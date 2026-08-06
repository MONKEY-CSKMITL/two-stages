"""
train.py — ตัวรันเทรนจริง อ่านค่าตั้งต้นทั้งหมดจากไฟล์ config

input:  configs/xxx.yaml
            ไฟล์ตั้งค่าที่ระบุทุกอย่าง: ใช้ข้อมูลชุดไหน, backbone อะไร, โจทย์ 4 คลาส/3class/binary,
            เทรนกี่รอบ ฯลฯ — เปลี่ยนการทดลองโดยแก้ไฟล์นี้ ไม่ต้องแก้โค้ด

        data/processed/splits/{variant}_{train,val,test}.csv
            ไฟล์ split ที่ run_split.py สร้างไว้ (อ่านตามที่ config ระบุ)

output: outputs/runs/{ชื่อการทดลอง}/best.pt
            weight ของโมเดลที่ทำคะแนนดีที่สุดบนชุด val

        outputs/runs/{ชื่อการทดลอง}/metrics.json
            ผลวัดบนชุด test + ตารางสับสน + ผลแยกตาม grade เดิม

        outputs/runs/{ชื่อการทดลอง}/config_used.yaml
            สำเนา config ที่ใช้จริงในรอบนี้ (ไว้ย้อนดูว่าผลนี้มาจากการตั้งค่าอะไร)

USAGE:
    python scripts/train.py --config configs/stage2_efficientnet_b0.yaml
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import yaml

# บอก Python ว่าให้หา module ในโฟลเดอร์ src/ ด้วย
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from core.stage2.dataset import (load_split_csv, VertebraDataset,
                                 compute_class_weights, TASKS,
                                 load_metadata, attach_metadata,
                                 compute_metadata_stats, NUM_METADATA)
from core.stage2.model import build_model, count_parameters
from core.stage2.losses import FocalLoss
from core.stage2.losses_ce import WeightedCrossEntropyLoss
from core.stage2.sampling import DownsampledNormalSampler
from core.utils.reporting import generate_all_reports


# ============================================================================
# ส่วนที่ 1: ตัววัดผล (metrics)
# ============================================================================

def compute_metrics(targets, preds, probs, num_classes):
    """
    คำนวณตัวชี้วัดผลจากคำตอบจริงกับคำตอบที่โมเดลทาย

    targets = คำตอบจริง (array ของเลขคลาส)
    preds   = คำตอบที่โมเดลทาย (array ของเลขคลาส)
    probs   = ความน่าจะเป็นของทุกคลาส รูปร่าง (จำนวนตัวอย่าง, จำนวนคลาส)
    """
    from sklearn.metrics import f1_score, roc_auc_score

    results = {}

    # macro F1 = เฉลี่ยคะแนน F1 ของทุกคลาสแบบให้น้ำหนักเท่ากัน
    # ใช้ "macro" ไม่ใช่ "weighted" เพราะข้อมูลไม่สมดุลมาก ถ้าใช้ weighted
    # คลาสปกติ (92%) จะกลบคะแนนคลาสอื่นจนดูดีเกินจริง
    results["macro_f1"] = float(f1_score(targets, preds, average="macro", zero_division=0))

    # AUC = ความสามารถในการแยกแยะระหว่างคลาส (0.5 = เดาสุ่ม, 1.0 = สมบูรณ์แบบ)
    try:
        if num_classes == 2:
            # โจทย์ 2 คลาส: ใช้ความน่าจะเป็นของคลาส "เสียหาย" อย่างเดียว
            results["auc"] = float(roc_auc_score(targets, probs[:, 1]))
        else:
            # โจทย์หลายคลาส: เทียบทีละคลาสกับที่เหลือ (one-vs-rest) แล้วเฉลี่ย
            results["auc"] = float(roc_auc_score(targets, probs, multi_class="ovr", average="macro"))
    except ValueError:
        # เกิดได้ถ้าชุดที่วัดผลไม่มีบางคลาสเลย (เช่น val set เล็กมาก)
        results["auc"] = float("nan")

    return results


def per_grade_recall(targets_4class, preds, task):
    """
    แยกดูว่าโมเดลจับ grade เดิมแต่ละระดับได้แค่ไหน

    ทำไมต้องมี: ตอนใช้โจทย์ที่ยุบคลาส (binary, 3class) โมเดลอาจจับ grade รุนแรง
    ได้หมดแต่พลาด grade เบาๆ ทั้งหมด แล้วยังได้คะแนนรวมดูดีอยู่ — ฟังก์ชันนี้เปิดโปง
    กรณีแบบนั้นให้เห็นว่าพลาด grade ไหนไปจริงๆ

    วิธีคิด: ใช้ label_map ของโจทย์นั้น (จาก TASKS) แปลง grade เดิม (0-3) เป็น
    "คำตอบที่ถูกต้องตามโจทย์นี้" ก่อน แล้วค่อยเทียบกับคำตอบที่โมเดลทาย — ใช้ label_map
    ตัวเดียวกับที่ dataset.py ใช้ตอนเตรียม label ตอนเทรน จึงรับประกันว่าตรงกันเสมอ
    ไม่ว่าจะเพิ่มโจทย์ใหม่กี่แบบก็ใช้ได้โดยไม่ต้องมาเขียน if/else เพิ่มทีละโจทย์

    targets_4class = grade เดิม 4 ระดับของทุกตัวอย่าง (0-3)
    preds          = คำตอบที่โมเดลทาย (ตามโจทย์ที่เลือก)
    """
    grade_names = ["normal", "mild", "moderate", "severe"]
    label_map = TASKS[task]["label_map"]   # เช่น 3class -> {"0":0,"1":1,"2":1,"3":2,"4":2}
    out = {}

    for g in range(4):
        mask = targets_4class == g              # เลือกเฉพาะตัวอย่างที่ grade เดิม = g
        n = int(mask.sum())
        if n == 0:
            continue                             # ไม่มีตัวอย่าง grade นี้เลย ข้ามไป

        # คำตอบที่ "ถูกต้อง" ของ grade นี้ ในพื้นที่ label ของโจทย์นี้ (ไม่ใช่ grade เดิมเสมอไป)
        expected_label = label_map[str(g)]
        correct = preds[mask] == expected_label

        out[grade_names[g]] = {
            "n": n,
            "recall": float(correct.sum() / n),   # สัดส่วนที่จับได้ถูก
        }

    return out


# ============================================================================
# ส่วนที่ 2: วนลูปเทรนและวัดผล
# ============================================================================

@torch.no_grad()   # บอก PyTorch ว่า "ตรงนี้ไม่ต้องเตรียมข้อมูลสำหรับ backprop" -> เร็วขึ้น ประหยัด memory
def evaluate(model, loader, device, criterion=None):
    """
    ให้โมเดลทายทั้งชุด แล้วรวบรวมผลกลับมา (ไม่มีการเรียนรู้เกิดขึ้นตรงนี้)

    criterion = ถ้าใส่มา จะคำนวณ loss ของชุดนี้ด้วย (ใช้ตอนวัด val_loss ราย epoch)
                ถ้าไม่ใส่ (None) จะคืน loss เป็น None

    คืนค่า dict เพื่อไม่ให้ต้องจำลำดับ tuple ยาวๆ (เดิมคืน 4 ค่า ตอนนี้ต้องการ 6)
    """
    model.eval()   # สลับโมเดลเป็นโหมด "วัดผล" — Dropout จะหยุดทำงาน (สำคัญมาก ไม่งั้นผลไม่นิ่ง)

    all_logits, all_targets, all_grade4, all_levels = [], [], [], []
    running_loss = 0.0

    # DataLoader คืนข้อมูลทีละก้อน (batch) — แต่ละก้อนมี 4 อย่างตามที่ dataset กำหนด
    for image, level_idx, metadata, label, grade_4class in loader:
        image = image.to(device)          # ย้ายข้อมูลไปที่ GPU (หรือ CPU ถ้าไม่มี GPU)
        level_idx = level_idx.to(device)
        metadata = metadata.to(device)

        logits = model(image, level_idx, metadata)   # ให้โมเดลทาย

        # คำนวณ loss ของชุดนี้ด้วย (ถ้าส่ง criterion มา)
        if criterion is not None:
            loss = criterion(logits, label.to(device))
            running_loss += loss.item() * image.size(0)   # คูณจำนวนรูปเพื่อถ่วงตามขนาด batch

        all_logits.append(logits.cpu())     # ย้ายผลกลับมา CPU เพื่อเก็บรวบรวม
        all_targets.append(label)
        all_grade4.append(grade_4class)
        all_levels.append(level_idx.cpu())

    # torch.cat = ต่อผลจากทุก batch เข้าด้วยกันเป็นก้อนเดียว
    logits = torch.cat(all_logits)
    targets = torch.cat(all_targets).numpy()
    grade4 = torch.cat(all_grade4).numpy()
    # +1 เพื่อแปลงกลับจาก 0-14 (ที่โมเดลใช้) เป็น 1-15 (level_index จริงที่คนอ่าน)
    levels = torch.cat(all_levels).numpy() + 1

    probs = torch.softmax(logits, dim=1).numpy()   # แปลงคะแนนดิบเป็นความน่าจะเป็น
    preds = probs.argmax(1)                         # เลือกคลาสที่ได้คะแนนสูงสุดเป็นคำตอบ

    avg_loss = running_loss / len(loader.dataset) if criterion is not None else None

    return {
        "targets": targets,
        "preds": preds,
        "probs": probs,
        "grade4": grade4,
        "levels": levels,
        "loss": avg_loss,
    }


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    เทรน 1 รอบ (1 epoch = เห็นข้อมูลทั้งชุดครบ 1 ครั้ง)
    คืนค่า loss เฉลี่ยของรอบนี้
    """
    model.train()   # สลับเป็นโหมด "เทรน" — Dropout เริ่มทำงาน
    running = 0.0

    for image, level_idx, metadata, label, _grade4 in loader:   # _grade4 = ใช้ตอนวัดผล ไม่ใช้ตอนเทรน
        image = image.to(device)
        level_idx = level_idx.to(device)
        metadata = metadata.to(device)
        label = label.to(device)

        # --- 4 บรรทัดนี้คือหัวใจของการเรียนรู้ทั้งหมด ---
        optimizer.zero_grad()                   # 1. ล้างค่าที่ค้างจากรอบก่อน (PyTorch สะสมค่าไว้โดยปริยาย)
        loss = criterion(model(image, level_idx, metadata), label)   # 2. ทาย แล้ววัดว่าผิดแค่ไหน
        loss.backward()                          # 3. คำนวณย้อนกลับว่าแต่ละ weight ควรปรับไปทางไหน
        optimizer.step()                         # 4. ปรับ weight จริงตามที่คำนวณได้

        # loss.item() = ดึงตัวเลขออกจาก tensor, คูณจำนวนรูปเพื่อถ่วงน้ำหนักตามขนาด batch
        running += loss.item() * image.size(0)

    return running / len(loader.dataset)


# ============================================================================
# ส่วนที่ 3: main — ประกอบทุกอย่างเข้าด้วยกัน
# ============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path, help="ไฟล์ config .yaml")
    args = ap.parse_args()

    # --- อ่าน config ---
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed = cfg["experiment"]["seed"]
    torch.manual_seed(seed)      # ล็อกการสุ่มของ PyTorch (การสุ่มค่าเริ่มต้น weight, dropout ฯลฯ)
    np.random.seed(seed)          # ล็อกการสุ่มของ numpy
    # ล็อก seed ทำให้รันซ้ำได้ผลเดิม — จำเป็นมากเวลาเทียบผลระหว่างการทดลอง
    # ไม่งั้นจะแยกไม่ออกว่าผลต่างเพราะสิ่งที่เราเปลี่ยน หรือเพราะการสุ่มต่างกัน

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    out_dir = Path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, out_dir / "config_used.yaml")   # เก็บสำเนา config ไว้ย้อนดู

    # --- โหลดข้อมูล ---
    task = cfg["data"]["task"]
    num_classes = TASKS[task]["num_classes"]
    class_names = TASKS[task]["class_names"]

    split_dir = Path(cfg["data"]["split_dir"])
    variant = cfg["data"]["variant"]

    # อ่านไฟล์ split ทั้ง 3 ชุด (ตรวจสอบข้อมูลอัตโนมัติในนี้ ถ้าผิดจะหยุดทันที)
    dfs = {}
    for split in ["train", "val", "test"]:
        dfs[split] = load_split_csv(str(split_dir / f"{variant}_{split}.csv"), task=task)
        n_patients = dfs[split]["patient_id"].nunique()
        dist = dfs[split]["label"].value_counts().reindex(range(num_classes)).fillna(0).astype(int)
        parts = "  ".join(f"{class_names[c]}={dist[c]}" for c in range(num_classes))
        print(f"  {split:5s}: {len(dfs[split]):5d} ปล้อง / {n_patients:3d} คน | {parts}")

    backbone = cfg["model"]["backbone"]
    img_size = cfg["data"].get("img_size")   # อาจเป็น None = ให้ transforms เลือกเอง
    batch_size = cfg["train"]["batch_size"]
    num_workers = cfg["train"].get("num_workers", 0)

    # --- metadata (ไม่บังคับ) ---
    # เปิดใช้เมื่อ config ระบุทั้ง data.metadata_xlsx และ model.use_metadata: true
    use_metadata = cfg["model"].get("use_metadata", False)
    metadata_stats = None

    if use_metadata:
        metadata_path = cfg["data"].get("metadata_xlsx")
        if not metadata_path:
            raise ValueError("ตั้ง model.use_metadata: true แล้ว ต้องระบุ data.metadata_xlsx ด้วย")

        print(f"\nโหลด metadata จาก {metadata_path}")
        metadata = load_metadata(metadata_path)

        # เชื่อม metadata เข้าทุก split
        for split in ["train", "val", "test"]:
            dfs[split] = attach_metadata(dfs[split], metadata)

        # คำนวณค่าสถิติสำหรับปรับสเกลจาก "ชุด train เท่านั้น"
        # ห้ามใช้ val/test เพราะจะเป็นการรั่วไหลข้อมูล (ค่าเฉลี่ยของชุดทดสอบ
        # จะแอบส่งผลต่อการเทรน ทำให้ผลดูดีเกินจริง)
        metadata_stats = compute_metadata_stats(dfs["train"])
        print("  ค่าสถิติที่ใช้ปรับสเกล (คำนวณจากชุด train เท่านั้น):")
        for col, s in metadata_stats.items():
            print(f"    {col:8s}: mean={s['mean']:.1f}  std={s['std']:.1f}  median={s['median']:.1f}")

    resize_mode = cfg["data"].get("resize_mode", "pad")   # "pad" (เดิม) หรือ "stretch"

    loaders = {}
    # downsample_ratio = สัดส่วน normal:fracture ที่ต้องการต่อ epoch (เช่น 5.0 = 5:1)
    # ไม่ใส่ (None, ค่าเริ่มต้น) = ไม่ downsample เลย พฤติกรรมเหมือนเดิมทุกประการ
    downsample_ratio = cfg["data"].get("downsample_ratio")

    for split in ["train", "val", "test"]:
        ds = VertebraDataset(dfs[split], backbone=backbone, img_size=img_size,
                             metadata_stats=metadata_stats, resize_mode=resize_mode)

        if split == "train" and downsample_ratio is not None:
            # ใช้ sampler แทน shuffle=True — สุ่มปล้อง normal ใหม่ทุก epoch ตาม ratio
            # (val/test ไม่ downsample เลย ต้องวัดผลบนสัดส่วนธรรมชาติของโลกจริงเสมอ)
            sampler = DownsampledNormalSampler(dfs["train"], ratio=downsample_ratio, seed=seed)
            loaders[split] = DataLoader(
                ds, batch_size=batch_size, sampler=sampler,   # sampler กับ shuffle ใช้พร้อมกันไม่ได้
                num_workers=num_workers, pin_memory=(device == "cuda"),
            )
        else:
            loaders[split] = DataLoader(
                ds,
                batch_size=batch_size,
                # shuffle เฉพาะชุด train — สลับลำดับทุก epoch ช่วยให้เรียนรู้ดีขึ้น
                # ชุด val/test ไม่ shuffle เพื่อให้ผลออกมาเรียงเหมือนเดิมทุกครั้ง (ตรวจสอบง่าย)
                shuffle=(split == "train"),
                num_workers=num_workers,
                pin_memory=(device == "cuda"),   # ช่วยให้ย้ายข้อมูลไป GPU เร็วขึ้น
            )

    # --- สร้างโมเดล ---
    model = build_model(
        backbone=backbone,
        num_classes=num_classes,
        level_embed_dim=cfg["model"].get("level_embed_dim", 32),
        pretrained=cfg["model"].get("pretrained", True),
        head_dropout=cfg["model"].get("head_dropout", 0.2),
        freeze_backbone=cfg["model"].get("freeze_backbone", False),
        use_metadata=use_metadata,
        num_metadata=NUM_METADATA,
        metadata_embed_dim=cfg["model"].get("metadata_embed_dim", 16),
    ).to(device)

    p = count_parameters(model)
    print(f"\nโมเดล: {backbone} | พารามิเตอร์ทั้งหมด {p['total']:,} "
          f"(เรียนรู้ได้ {p['trainable']:,} / ล็อกไว้ {p['frozen']:,})")

    # --- loss + optimizer ---
    # น้ำหนักถ่วงคลาสคำนวณจากชุด train เท่านั้น (ห้ามใช้ val/test เด็ดขาด
    # เพราะจะเป็นการแอบเอาข้อมูลชุดวัดผลมาใช้ตอนเทรน = ผลลัพธ์ไม่น่าเชื่อถือ)
    alpha = None
    if cfg["loss"].get("use_class_weights", True):
        alpha = compute_class_weights(dfs["train"], num_classes).to(device)
        print(f"น้ำหนักถ่วงคลาส: {[round(float(w), 2) for w in alpha]}")

    # เลือก loss function จาก config — ค่าเริ่มต้น "focal" เพื่อให้ config เก่าทุกไฟล์
    # ที่ไม่มีบรรทัด loss.type ระบุไว้ (รันมาแล้วก่อนหน้านี้ทั้งหมด) ยังทำงานเหมือนเดิม
    # ทุกประการ ไม่กระทบผลที่มีอยู่แล้วเลย
    #
    # "ce" (Weighted Cross Entropy) เหมาะกับตอนที่ downsample ข้อมูลให้สมดุลแล้ว
    # (ไม่จำเป็นต้องมีกลไก "โฟกัสตัวอย่างยาก" ของ Focal Loss ซ้อนอีกชั้น)
    loss_type = cfg["loss"].get("type", "focal")
    if loss_type == "focal":
        criterion = FocalLoss(alpha=alpha, gamma=cfg["loss"].get("gamma", 2.0))
    elif loss_type == "ce":
        criterion = WeightedCrossEntropyLoss(alpha=alpha)
    else:
        raise ValueError(f"loss.type ต้องเป็น 'focal' หรือ 'ce' เท่านั้น ได้รับ '{loss_type}'")
    print(f"loss function: {loss_type}")

    # AdamW = อัลกอริทึมปรับ weight ที่นิยมใช้กับงานภาพ (ปรับความเร็วการเรียนรู้ให้แต่ละ weight เอง)
    optimizer = torch.optim.AdamW(
        # filter(...) = ส่งเฉพาะพารามิเตอร์ที่เรียนรู้ได้ให้ optimizer
        # จำเป็นเมื่อ freeze_backbone=True (ไม่งั้น optimizer จะพยายามปรับตัวที่ล็อกไว้)
        filter(lambda q: q.requires_grad, model.parameters()),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"].get("weight_decay", 1e-4),
    )

    epochs = cfg["train"]["epochs"]
    # scheduler = ตัวค่อยๆ ลดความเร็วการเรียนรู้ลงเมื่อใกล้จบ
    # ช่วงแรกเรียนเร็ว (ปรับเยอะ) ช่วงท้ายเรียนช้า (ปรับละเอียด) — ช่วยให้ผลนิ่งขึ้น
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # --- วนลูปเทรน ---
    patience = cfg["train"].get("patience", 5)
    best_score, best_epoch, bad_epochs = -1.0, -1, 0

    print(f"\nเริ่มเทรน {epochs} รอบ (หยุดก่อนถ้าไม่ดีขึ้นติดกัน {patience} รอบ)\n")

    # เก็บค่าราย epoch ไว้ทำกราฟตอนจบ
    history_rows = []

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, loaders["train"], criterion, optimizer, device)
        scheduler.step()

        # วัดผลบนชุด val ทุกรอบ เพื่อตัดสินใจว่าจะเก็บ weight รอบนี้ไหม / จะหยุดหรือยัง
        # ส่ง criterion ไปด้วยเพื่อให้คำนวณ val_loss (เอาไว้ทำกราฟเทียบกับ train_loss)
        val_out = evaluate(model, loaders["val"], device, criterion=criterion)
        val_metrics = compute_metrics(val_out["targets"], val_out["preds"],
                                      val_out["probs"], num_classes)

        print(f"epoch {epoch:02d}  train_loss={train_loss:.4f}  val_loss={val_out['loss']:.4f}  "
              f"val_F1={val_metrics['macro_f1']:.4f}  val_AUC={val_metrics['auc']:.4f}")

        history_rows.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_out["loss"],
            "val_f1": val_metrics["macro_f1"],
            "val_auc": val_metrics["auc"],
        })

        # ใช้ macro F1 บนชุด val เป็นตัวตัดสินว่ารอบไหนดีที่สุด
        # (ไม่ใช้ accuracy เพราะข้อมูลไม่สมดุล — ทายปกติหมดก็ได้ accuracy สูงแต่ไร้ประโยชน์)
        score = val_metrics["macro_f1"]
        if score > best_score:
            best_score, best_epoch, bad_epochs = score, epoch, 0
            torch.save(model.state_dict(), out_dir / "best.pt")   # เก็บเฉพาะ weight ที่ดีที่สุด
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                # หยุดก่อนกำหนด (early stopping) — เทรนต่อไปมีแต่จะ overfit
                print(f"\nหยุดก่อนกำหนดที่รอบ {epoch} (รอบที่ดีที่สุดคือ {best_epoch})")
                break

    history = pd.DataFrame(history_rows)

    # --- วัดผลครั้งสุดท้ายบนชุด test ด้วย weight ที่ดีที่สุด ---
    # ใช้ชุด test แค่ครั้งเดียวตอนจบเท่านั้น ไม่ใช้ระหว่างเทรน
    # (ถ้าใช้ระหว่างทางเพื่อตัดสินใจอะไร ชุด test จะกลายเป็นชุดปรับจูนไปโดยปริยาย
    #  ทำให้ตัวเลขสุดท้ายดูดีเกินความจริง)
    model.load_state_dict(torch.load(out_dir / "best.pt", map_location=device))
    test_out = evaluate(model, loaders["test"], device)

    targets, preds = test_out["targets"], test_out["preds"]
    test_metrics = compute_metrics(targets, preds, test_out["probs"], num_classes)
    grade_breakdown = per_grade_recall(test_out["grade4"], preds, task)

    from sklearn.metrics import confusion_matrix, classification_report
    cm = confusion_matrix(targets, preds, labels=list(range(num_classes)))

    print("\n" + "=" * 60)
    print("ผลบนชุด TEST")
    print("=" * 60)
    for k, v in test_metrics.items():
        print(f"  {k:10s} = {v:.4f}")
    print()
    print(classification_report(targets, preds, target_names=class_names,
                                labels=list(range(num_classes)), zero_division=0))
    print("ตารางสับสน (แถว=จริง, คอลัมน์=ทาย):")
    print(cm)
    print("\nผลแยกตาม grade เดิม (ดูว่าจับ 'เล็กน้อย' ได้จริงไหม):")
    for g, info in grade_breakdown.items():
        print(f"  {g:9s}: n={info['n']:4d}  recall={info['recall']:.3f}")

    # --- สร้างกราฟและตารางทั้งหมด ---
    df_overall, df_per_level = generate_all_reports(
        history, targets, preds, test_out["levels"], class_names, out_dir)

    print("\nตาราง metric รวมทุกปล้อง:")
    print(df_overall.to_string(index=False))

    # --- บันทึกผลลงไฟล์ ---
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({
            "experiment": cfg["experiment"]["name"],
            "backbone": backbone,
            "task": task,
            "variant": variant,
            "use_metadata": use_metadata,
            "best_epoch": best_epoch,
            "test_metrics": test_metrics,
            "per_grade_recall": grade_breakdown,
            "confusion_matrix": cm.tolist(),
        }, f, indent=2, ensure_ascii=False)

    print(f"\nบันทึกผลไว้ที่ {out_dir}")
    print(f"  กราฟ  -> {out_dir / 'plots'}")
    print(f"  ตาราง -> {out_dir / 'tables'}")
    print(f"  ค่าราย epoch -> {out_dir / 'history.csv'}")


if __name__ == "__main__":
    main()