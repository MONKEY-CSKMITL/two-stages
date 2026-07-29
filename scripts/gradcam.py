"""
gradcam.py — สร้างภาพ heatmap แสดงว่าโมเดล "มองตรงไหน" ตอนตัดสินใจ

input:  --config configs/xxx.yaml
            ไฟล์ config เดียวกับที่ใช้เทรน (จะไปอ่าน best.pt จาก output.dir ที่ระบุไว้)
        outputs/runs/{ชื่อ}/best.pt
            weight ที่เทรนเสร็จแล้ว

output: outputs/runs/{ชื่อ}/plots/gradcam_grid.png
            ตารางภาพ heatmap แยกตาม grade จริง (แถว) x ตัวอย่าง (คอลัมน์)

ทำไมต้องดู:
    ตัวเลข macro F1 บอกแค่ "ทายถูกกี่ %" แต่ไม่บอกว่า "ทายถูกด้วยเหตุผลอะไร"
    โมเดลอาจทายถูกเพราะจับสัญญาณที่ไม่เกี่ยวข้อง (เช่น ขอบดำที่เกิดจากการตัดภาพ
    แบบ masked) แทนที่จะดูเนื้อกระดูกจริง — heatmap เปิดโปงกรณีแบบนั้นได้

    สีแดง/เหลือง = บริเวณที่มีอิทธิพลต่อการตัดสินใจมาก
    สีน้ำเงิน    = บริเวณที่แทบไม่มีผล

วิธีรัน:
    pip install grad-cam
    python scripts/gradcam.py --config configs/stage2_effb0_masked.yaml
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

import matplotlib
matplotlib.use("Agg")   # โหมดไม่เปิดหน้าต่าง (จำเป็นเวลารันผ่าน terminal)
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from core.stage2.dataset import (load_split_csv, VertebraDataset, TASKS,
                                 load_metadata, attach_metadata,
                                 compute_metadata_stats, NUM_METADATA)
from core.stage2.model import build_model
from core.stage2.transforms import IMAGENET_MEAN, IMAGENET_STD

LEVEL_NAMES = ["T3", "T4", "T5", "T6", "T7", "T8", "T9", "T10",
               "T11", "T12", "L1", "L2", "L3", "L4", "L5"]


class SingleInputWrapper(torch.nn.Module):
    """
    ตัวห่อโมเดลให้รับ "รูปอย่างเดียว"

    ทำไมต้องมี: โมเดลของเรารับ 3 อย่าง (รูป, ปล้องไหน, metadata) แต่ library
    Grad-CAM ถูกออกแบบมาสำหรับโมเดลที่รับแค่รูป — ตัวห่อนี้จะ "ตรึง" ค่าปล้องกับ
    metadata ไว้คงที่ แล้วเปิดให้เฉพาะรูปเป็นตัวแปรที่ Grad-CAM จะวิเคราะห์
    (ซึ่งถูกต้องตามที่เราต้องการ เพราะเราอยากรู้ว่า "ในรูป" ตรงไหนสำคัญ)
    """

    def __init__(self, model, level_idx: int, metadata: torch.Tensor, device):
        super().__init__()
        self.model = model
        self.level_idx = level_idx
        self.metadata = metadata     # (1, NUM_METADATA)
        self.device = device

    def forward(self, image):
        n = image.shape[0]
        lvl = torch.full((n,), self.level_idx, dtype=torch.long, device=self.device)
        meta = self.metadata.expand(n, -1).to(self.device)
        return self.model(image, lvl, meta)


def denormalize(img_tensor: torch.Tensor) -> np.ndarray:
    """
    แปลงรูปที่ normalize แล้วกลับเป็นค่า 0.0-1.0 เพื่อเอาไปวาดให้ตาคนดูได้

    (ตอนเตรียมรูปเราลบค่าเฉลี่ยและหารส่วนเบี่ยงเบนมาตรฐานไป ทำให้มีค่าติดลบ
    วาดเป็นภาพไม่ได้ ต้องย้อนกลับก่อน)
    """
    img = img_tensor.detach().cpu().numpy().transpose(1, 2, 0)   # (C,H,W) -> (H,W,C)
    img = img * IMAGENET_STD + IMAGENET_MEAN                       # ย้อนสูตร normalize
    return np.clip(img, 0, 1)


def pick_samples(df, n_per_grade: int, seed: int = 42):
    """
    เลือกตัวอย่างมาทำ heatmap — เอา n ตัวต่อ grade จริงแต่ละระดับ

    เลือกแบบสุ่ม (ล็อก seed) แทนที่จะเอาตัวที่โมเดลมั่นใจสุด เพราะอยากเห็น
    พฤติกรรมทั่วไป ไม่ใช่เฉพาะเคสที่ดูดี
    """
    rng = np.random.RandomState(seed)
    picked = []
    for g in range(4):
        idx = df.index[df["grade_4class"] == g].tolist()
        if not idx:
            continue
        take = rng.choice(idx, size=min(n_per_grade, len(idx)), replace=False)
        picked.extend([(g, int(i)) for i in take])
    return picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--n_per_grade", type=int, default=3,
                    help="จำนวนตัวอย่างต่อ grade (default 3 -> รวม 12 ภาพ)")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"],
                    help="ใช้ข้อมูลชุดไหน (default test)")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    backbone = cfg["model"]["backbone"]

    # Grad-CAM แบบมาตรฐานใช้ได้กับ CNN เท่านั้น — ViT ต้องใช้เทคนิคคนละแบบ
    # (attention rollout) ซึ่งยังไม่ได้ทำ จึงหยุดพร้อมบอกเหตุผลชัดเจน
    # ดีกว่าปล่อยให้รันแล้วได้ภาพที่ไม่มีความหมาย
    if backbone == "rad_dino":
        raise NotImplementedError(
            "Grad-CAM แบบนี้ใช้กับ CNN (EfficientNet, ConvNeXt) เท่านั้น\n"
            "rad_dino เป็น Vision Transformer ซึ่งไม่มีชั้น conv ให้จับ heatmap ตรงๆ\n"
            "ต้องใช้เทคนิคคนละแบบ (attention rollout) ซึ่งยังไม่ได้ทำในสคริปต์นี้"
        )

    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    except ImportError:
        raise ImportError("ต้องติดตั้งก่อน: pip install grad-cam")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(cfg["output"]["dir"])
    ckpt_path = out_dir / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"ไม่เจอไฟล์ weight ที่ {ckpt_path} — เทรนให้เสร็จก่อน")

    task = cfg["data"]["task"]
    num_classes = TASKS[task]["num_classes"]
    class_names = TASKS[task]["class_names"]
    grade_names = ["normal", "mild", "moderate", "severe"]

    # --- โหลดข้อมูล (ใช้ค่าตั้งเดียวกับตอนเทรนทุกอย่าง) ---
    split_dir = Path(cfg["data"]["split_dir"])
    variant = cfg["data"]["variant"]
    df = load_split_csv(str(split_dir / f"{variant}_{args.split}.csv"), task=task)

    use_metadata = cfg["model"].get("use_metadata", False)
    metadata_stats = None
    if use_metadata:
        metadata = load_metadata(cfg["data"]["metadata_xlsx"])
        df = attach_metadata(df, metadata)
        # ต้องใช้ค่าสถิติจากชุด train เหมือนตอนเทรน ไม่ใช่จากชุดที่กำลังดู
        train_df = load_split_csv(str(split_dir / f"{variant}_train.csv"), task=task)
        train_df = attach_metadata(train_df, metadata)
        metadata_stats = compute_metadata_stats(train_df)

    ds = VertebraDataset(df, backbone=backbone,
                         img_size=cfg["data"].get("img_size"),
                         metadata_stats=metadata_stats)

    # --- โหลดโมเดลที่เทรนแล้ว ---
    model = build_model(
        backbone=backbone,
        num_classes=num_classes,
        level_embed_dim=cfg["model"].get("level_embed_dim", 32),
        pretrained=False,                # ไม่ต้องโหลด weight เริ่มต้น เพราะจะทับด้วย best.pt อยู่แล้ว
        head_dropout=cfg["model"].get("head_dropout", 0.2),
        use_metadata=use_metadata,
        num_metadata=NUM_METADATA,
        metadata_embed_dim=cfg["model"].get("metadata_embed_dim", 16),
    ).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    # conv_head = ชั้น conv สุดท้ายของ EfficientNet ที่ยังมีมิติเชิงพื้นที่เหลืออยู่ (7x7)
    # ต้องใช้ชั้นที่ยังมี H,W ไม่งั้นวาด heatmap ทับรูปไม่ได้
    target_layer = model.backbone.conv_head

    picked = pick_samples(df, args.n_per_grade)
    print(f"เลือกตัวอย่าง {len(picked)} ภาพจากชุด {args.split}")

    # --- ทำ Grad-CAM ทีละภาพ ---
    results = []
    for true_grade, idx in picked:
        image, level_idx, meta, label, grade4 = ds[idx]
        image = image.unsqueeze(0).to(device)     # (1,3,H,W)
        meta = meta.unsqueeze(0)

        wrapped = SingleInputWrapper(model, level_idx, meta, device).to(device).eval()

        # ให้โมเดลทายก่อน เพื่อรู้ว่าจะอธิบายคำตอบไหน
        with torch.no_grad():
            pred = int(wrapped(image).argmax(1).item())

        # อธิบาย "คำตอบที่โมเดลเลือกเอง" (ไม่ใช่คำตอบที่ถูก) เพราะเราอยากรู้ว่า
        # ทำไมมันถึงตอบแบบนั้น ไม่ว่าจะถูกหรือผิด
        with GradCAM(model=wrapped, target_layers=[target_layer]) as cam:
            grayscale = cam(input_tensor=image,
                            targets=[ClassifierOutputTarget(pred)])[0]

        rgb = denormalize(image[0])
        overlay = show_cam_on_image(rgb, grayscale, use_rgb=True)

        results.append({
            "overlay": overlay,
            "level": LEVEL_NAMES[level_idx],
            "true_grade": grade_names[true_grade],
            "pred": class_names[pred],
            "correct": (pred == label),
        })

    # --- วาดตาราง: แถว = grade จริง, คอลัมน์ = ตัวอย่าง ---
    n_rows = len(set(r["true_grade"] for r in results))
    n_cols = args.n_per_grade
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 3.5 * n_rows))
    axes = np.array(axes).reshape(n_rows, n_cols)

    for i, r in enumerate(results):
        ax = axes[i // n_cols, i % n_cols]
        ax.imshow(r["overlay"])
        # กรอบเขียว = ทายถูก, กรอบแดง = ทายผิด (ดูปราดเดียวรู้)
        color = "green" if r["correct"] else "red"
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(3)
        ax.set_title(f"{r['level']} | true={r['true_grade']}\npred={r['pred']}",
                     fontsize=9, color=color)
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle(f"Grad-CAM — {cfg['experiment']['name']} ({args.split} set)\n"
                 f"red/yellow = high influence on the decision", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    save_path = plots_dir / f"gradcam_{args.split}.png"
    fig.savefig(save_path, dpi=140)
    plt.close(fig)

    n_correct = sum(r["correct"] for r in results)
    print(f"ทายถูก {n_correct}/{len(results)} ภาพในกลุ่มตัวอย่างนี้")
    print(f"บันทึกภาพไว้ที่ {save_path}")


if __name__ == "__main__":
    main()