"""
build_augmented_trainset.py — ขยายชุด train ด้วย augmentation แบบ "เขียนลงไฟล์จริง"

--------------------------------------------------------------------------
ต่างจาก data.augment ใน config อย่างไร — ต้องเข้าใจก่อนใช้
--------------------------------------------------------------------------
                      data.augment (เดิม)        ไฟล์นี้ (offline)
  ไฟล์บนดิสก์          8,409 ใบ                   84,090 ใบ (ที่ factor 10)
  สุ่มตอนไหน           ทุกครั้งที่หยิบภาพ          ครั้งเดียวตอนสร้าง แล้วแช่แข็ง
  เวอร์ชันต่อ 1 ภาพ     ไม่จำกัด (ใหม่ทุก epoch)   ตายตัวเท่ากับ factor
  ชี้ให้คนอื่นดูได้ไหม   ไม่ได้                     ได้ เปิดโฟลเดอร์ดูได้เลย

ข้อเท็จจริงที่ต้องรู้: ที่จำนวน gradient step เท่ากัน แบบ offline ให้ความหลากหลาย
**น้อยกว่า** แบบ on-the-fly (ภาพใบเดิมมีได้แค่ N หน้าตา ไม่ใช่ใหม่ทุกรอบ) แต่แบบ
offline นับได้ ตรวจสอบได้ และรายงานเป็นตัวเลขในเล่มได้ ซึ่งเป็นคนละคุณค่ากัน

--------------------------------------------------------------------------
กับดัก 3 ข้อที่ไฟล์นี้ออกแบบมาเพื่อกันโดยเฉพาะ
--------------------------------------------------------------------------
1. **จำนวนรอบเฟ้อ 10 เท่าโดยไม่รู้ตัว** — 1 epoch บนข้อมูล 84,090 ใบ = 10 epoch เดิม
   ถ้ารัน 15 epoch เหมือนเดิมแล้วเอาไปเทียบตารางเก่า ผลที่ได้จะแปลว่า "เทรนนานกว่า
   10 เท่า" ไม่ใช่ "การขยายข้อมูลช่วย" — ไฟล์นี้จึงพิมพ์ epoch ที่เทียบเท่าให้ดูตอนจบ

2. **val/test ต้องไม่ถูก augment** แต่ก็ต้องไม่เป็นคนละ distribution กับ train ด้วย
   จึงเขียน val/test ออกมาเหมือนกันทุกอย่าง (ผ่าน preprocess + pad ขนาดเดียวกัน)
   ต่างกันแค่ **ไม่ augment และไม่คูณ** (1 เท่าเสมอ ไม่ว่าจะตั้ง factor เท่าไหร่)

3. **augment ซ้ำสองชั้น** — ไฟล์ที่ได้ผ่านท่อมาครบแล้ว ถ้า config ที่เอาไปใช้ยังตั้ง
   data.augment ไว้อีก ภาพจะโดน augment ทับอีกรอบ ไฟล์นี้จึงเขียน config ตัวอย่าง
   ที่ตั้ง preprocess/augment เป็น none ไว้ให้ด้วย

--------------------------------------------------------------------------
ลำดับที่ bake ลงไฟล์ — ตรงกับ transforms.py เป๊ะ
--------------------------------------------------------------------------
    เปิดไฟล์ -> preprocess -> resize/pad 224 -> augment -> เซฟ

augment อยู่หลัง pad โดยตั้งใจ เพราะ crop ตัดชิดตัวกระดูก ถ้าหมุนตอนยังไม่ pad
มุมกระดูกจะถูกตัดหายทันที (เหตุผลเดียวกับใน transforms.py)

ผลที่ได้เป็น variant ใหม่ ซึ่ง train.py ใช้ได้ทันทีโดยไม่ต้องแก้โค้ดสักบรรทัด
เพราะ train.py ประกอบ path เป็น {split_dir}/{variant}_{split}.csv อยู่แล้ว

USAGE:
    python scripts/build_augmented_trainset.py --factor 10
    python scripts/build_augmented_trainset.py --factor 10 --preprocess none --augment standard
    python scripts/build_augmented_trainset.py --class_factors "0:2,1:20,2:16,3:20"
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from core.stage2.augment import get_augment_fn
from core.stage2.dataset import load_split_csv
from core.stage2.preprocessing import get_preprocess_fn

GRADE_NAMES = {0: "normal", 1: "mild", 2: "moderate", 3: "severe"}

CONFIG_TEMPLATE = """# สร้างอัตโนมัติโดย scripts/build_augmented_trainset.py — แก้ได้ตามสบาย
#
# สำคัญ: variant นี้เป็นภาพที่ผ่าน preprocess + augment มาแล้วตั้งแต่ตอนสร้างไฟล์
# จึงต้องตั้ง data.preprocess และ data.augment เป็น none ทั้งคู่ ไม่งั้นภาพจะโดน
# แต่งซ้ำสองชั้น (ซึ่งไม่ใช่สิ่งที่ต้องการวัด)
#
# ชุด train มี {n_train:,} แถว = {factor}x ของเดิม ({n_orig:,})
# ฉะนั้น 1 epoch ที่นี่ = {factor} epoch ของชุดเดิม
# ตั้ง epochs = {epochs} -> เท่ากับ ~{equiv} epoch ของชุดเดิมในแง่จำนวน gradient step
# เวลาเทียบกับตารางผลเดิม (ซึ่งรันที่ 15 epoch) ต้องระบุตัวเลขนี้กำกับไว้ด้วยเสมอ

experiment:
  name: {name}
  seed: 42

data:
  split_dir: data/processed/splits
  variant: {variant}
  task: multiclass
  img_size: null
  resize_mode: pad
  preprocess: none
  augment: none

model:
  backbone: efficientnet_b0
  pretrained: true
  level_embed_dim: 32
  head_dropout: 0.2
  freeze_backbone: false

loss:
  gamma: 2.0
  use_class_weights: true

train:
  epochs: {epochs}
  batch_size: 64
  lr: 1.0e-4
  weight_decay: 1.0e-4
  patience: {patience}
  num_workers: 4

output:
  dir: outputs/runs/{name}
"""


def resize_pad(img: Image.Image, size: int) -> Image.Image:
    """ย่อ+เติมขอบดำแบบเดียวกับ _prepare_standard(resize_mode="pad") ใน transforms.py"""
    w, h = img.size
    scale = size / max(w, h)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    small = img.resize((new_w, new_h))
    canvas = Image.new("L", (size, size), 0)
    canvas.paste(small, ((size - new_w) // 2, (size - new_h) // 2))
    return canvas


def parse_class_factors(spec: str) -> dict:
    """แปลง "0:2,1:20,2:16,3:20" เป็น {0: 2, 1: 20, 2: 16, 3: 20}"""
    out = {}
    for part in spec.split(","):
        k, v = part.split(":")
        out[int(k.strip())] = int(v.strip())
    return out


def build_split(df: pd.DataFrame, split: str, out_root: Path, size: int,
                preprocess_fn, augment_fn, factors: dict, seed: int,
                overwrite: bool) -> pd.DataFrame:
    """
    เขียนไฟล์ภาพของ 1 split แล้วคืนตารางแถวใหม่

    factors = จำนวนสำเนาต่อ 1 ปล้อง แยกตาม grade
              สำเนาที่ 0 คือ "ต้นฉบับ" (ผ่าน preprocess + pad แต่ไม่ augment) เสมอ
              สำเนาที่ 1 ขึ้นไปถึงจะโดน augment — ทำแบบนี้เพื่อให้ข้อมูลจริงยังอยู่ครบ
              ไม่ใช่ถูกแทนที่ด้วยเวอร์ชันที่ถูกดัดแปลงทั้งหมด
    """
    rows = []
    t0 = time.time()
    n_written = n_skipped = 0

    for i, row in enumerate(df.itertuples(index=False)):
        src = Path(row.crop_path)
        pid = str(row.patient_id)
        dst_dir = out_root / pid
        dst_dir.mkdir(parents=True, exist_ok=True)

        # เตรียมภาพฐาน (preprocess + pad) ครั้งเดียว แล้วใช้ร่วมกันทุกสำเนา
        # การทำ preprocess ซ้ำทุกสำเนาได้ผลเหมือนกันเป๊ะอยู่แล้ว (deterministic)
        # แต่เสียเวลาเปล่า โดยเฉพาะ destripe ที่มี median filter ทีละแถว
        base = resize_pad(preprocess_fn(Image.open(src).convert("L")) if preprocess_fn
                          else Image.open(src).convert("L"), size)

        n_copies = factors.get(int(row.grade_4class), 1)
        for k in range(n_copies):
            dst = dst_dir / f"{src.stem}_a{k:02d}.png"

            if dst.exists() and not overwrite:
                n_skipped += 1
            else:
                if k == 0 or augment_fn is None:
                    img = base                     # สำเนาที่ 0 = ต้นฉบับ ไม่ augment
                else:
                    # seed ผูกกับ (แถว, สำเนาที่เท่าไหร่) — รันซ้ำได้ภาพเดิมเป๊ะ
                    # และสำเนาคนละใบของปล้องเดียวกันได้เลขสุ่มคนละชุดแน่นอน
                    np.random.seed((seed * 1_000_003 + i * 97 + k) % (2**31 - 1))
                    img = augment_fn(base)
                img.save(dst, format="PNG", optimize=True)
                n_written += 1

            rows.append({
                "patient_id": pid,
                "level_index": int(row.level_index),
                "level_name": row.level_name,
                "crop_path": str(dst.resolve()),
                "grade_raw": row.grade_raw,
                "aug_index": k,
                "source_crop_path": str(src),
            })

        if (i + 1) % 500 == 0:
            done = (i + 1) / len(df)
            el = time.time() - t0
            print(f"    {split}: {i + 1:,}/{len(df):,} ปล้อง ({done * 100:.0f}%) "
                  f"— เขียนแล้ว {n_written:,} ไฟล์ — เหลืออีกราว {el / done - el:.0f} วิ")

    print(f"    {split}: เสร็จ — {len(rows):,} แถว "
          f"(เขียนใหม่ {n_written:,} / ข้ามของเดิม {n_skipped:,}) "
          f"ใช้เวลา {time.time() - t0:.0f} วิ")
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split_dir", type=Path, default=Path("data/processed/splits"))
    ap.add_argument("--variant", default="xray_masked")
    ap.add_argument("--preprocess", default="destripe",
                    help="ชื่อจาก PREPROCESS_FNS — bake ลงไฟล์ทั้ง train/val/test")
    ap.add_argument("--augment", default="strong",
                    help="ชื่อจาก AUGMENT_FNS — bake ลงไฟล์ **เฉพาะ train**")
    ap.add_argument("--factor", type=int, default=10,
                    help="จำนวนสำเนาต่อ 1 ปล้อง (รวมต้นฉบับ) — ใช้กับทุกคลาสเท่ากัน")
    ap.add_argument("--class_factors", default=None,
                    help='ตั้งแยกรายคลาส เช่น "0:2,1:20,2:16,3:20" (ทับค่า --factor)')
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--crops_root", type=Path, default=Path("data/interim/crops_aug"))
    ap.add_argument("--epochs", type=int, default=None,
                    help="epochs ที่จะใส่ใน config ที่สร้างให้ — ไม่ระบุ = คำนวณให้เทียบเท่า 15 epoch เดิม")
    ap.add_argument("--patience", type=int, default=3, help="patience ที่จะใส่ใน config")
    ap.add_argument("--overwrite", action="store_true", help="เขียนทับไฟล์เดิมที่มีอยู่")
    ap.add_argument("--dry_run", action="store_true", help="คำนวณและพิมพ์แผนอย่างเดียว ไม่เขียนไฟล์")
    args = ap.parse_args()

    preprocess_fn = get_preprocess_fn(None if args.preprocess == "none" else args.preprocess)
    augment_fn = get_augment_fn(None if args.augment == "none" else args.augment)

    if args.class_factors:
        factors = parse_class_factors(args.class_factors)
        # ใส่ตัวคูณของทุกคลาสไว้ในชื่อ variant เลย เพราะเป็นข้อมูลที่ต้องรู้ตอนอ่านผล
        # ถ้าย่อเป็น "clsbalanced" จะกลับไปหาไม่ได้ว่าคลาสไหนคูณเท่าไหร่
        tag = "cls" + "-".join(str(factors.get(g, 1)) for g in [0, 1, 2, 3])
    else:
        factors = {g: args.factor for g in [0, 1, 2, 3]}
        tag = f"x{args.factor}"

    new_variant = f"{args.variant}_{args.preprocess}_{args.augment}_{tag}"
    out_root = args.crops_root / new_variant

    print("=" * 78)
    print(f"variant ใหม่ : {new_variant}")
    print(f"  preprocess = {args.preprocess}  (ใช้กับ train + val + test)")
    print(f"  augment    = {args.augment}  (ใช้กับ train เท่านั้น สำเนาที่ 1 ขึ้นไป)")
    print(f"  สำเนาต่อปล้อง = {factors}  (สำเนาที่ 0 = ต้นฉบับ ไม่ augment)")
    print(f"  ขนาดภาพ    = {args.size}x{args.size} (pad แล้ว)")
    print(f"  ปลายทาง    = {out_root}")
    print("=" * 78)

    # --- โหลดทั้ง 3 split ก่อน เพื่อพิมพ์แผนให้ดูก่อนลงมือเขียนไฟล์ ---
    dfs = {}
    for split in ["train", "val", "test"]:
        dfs[split] = load_split_csv(str(args.split_dir / f"{args.variant}_{split}.csv"),
                                    task="multiclass")

    print("\nแผนการขยายข้อมูล (ชุด train):")
    print(f"  {'grade':<12}{'เดิม':>9}{'x':>5}{'ใหม่':>10}")
    total_before = total_after = 0
    for g in [0, 1, 2, 3]:
        n = int((dfs["train"]["grade_4class"] == g).sum())
        f = factors.get(g, 1)
        print(f"  {GRADE_NAMES[g]:<12}{n:>9,}{f:>5}{n * f:>10,}")
        total_before += n
        total_after += n * f
    print(f"  {'รวม':<12}{total_before:>9,}{'':>5}{total_after:>10,}")

    ratio_before = (dfs["train"]["grade_4class"] == 0).sum() / max(
        (dfs["train"]["grade_4class"] == 1).sum(), 1)
    ratio_after = ((dfs["train"]["grade_4class"] == 0).sum() * factors.get(0, 1)) / max(
        (dfs["train"]["grade_4class"] == 1).sum() * factors.get(1, 1), 1)
    print(f"\n  สัดส่วน normal:mild  {ratio_before:.1f}:1  ->  {ratio_after:.1f}:1")
    if abs(ratio_after - ratio_before) < 0.05:
        print("  (ไม่เปลี่ยน — การคูณทุกคลาสเท่ากันไม่ได้แก้ความไม่สมดุล ใช้ --class_factors ถ้าต้องการแก้)")

    # จำนวน gradient step ต่อ epoch โตตาม "จำนวนแถวรวม" ไม่ใช่ตัวคูณของคลาสใดคลาสหนึ่ง
    # (ตอนคูณเฉพาะคลาสน้อย ตัวคูณรวมจะน้อยกว่าตัวคูณรายคลาสมาก เพราะคลาส normal
    #  ครองจำนวนแถวเกือบทั้งหมดอยู่แล้ว)
    expansion = total_after / total_before
    epoch_equiv = max(1, round(15 / expansion))
    print(f"\n  1 epoch บนชุดใหม่ = {expansion:.2f} epoch ของชุดเดิม")
    print(f"  -> ถ้าจะเทียบกับตาราง 15 epoch เดิมอย่างแฟร์ ต้องรันราว {epoch_equiv} epoch")

    est_gb = total_after * 13_000 / 1e9
    print(f"\n  ประมาณพื้นที่ดิสก์ที่ใช้: {est_gb:.2f} GB")

    if args.dry_run:
        print("\n--dry_run: ไม่ได้เขียนไฟล์อะไรเลย")
        return

    print("\nเริ่มเขียนไฟล์...")
    out_root.mkdir(parents=True, exist_ok=True)
    args.split_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train", "val", "test"]:
        # val/test: 1 สำเนาเสมอ และไม่ augment — คุมด้วยการ "ไม่ส่ง augment_fn มา"
        # ที่ตรงนี้จุดเดียว ทำให้เผลอ augment ชุดวัดผลไม่ได้เลยโดยโครงสร้าง
        split_factors = factors if split == "train" else {g: 1 for g in [0, 1, 2, 3]}
        split_aug = augment_fn if split == "train" else None

        out = build_split(dfs[split], split, out_root, args.size,
                          preprocess_fn, split_aug, split_factors, args.seed, args.overwrite)

        csv_path = args.split_dir / f"{new_variant}_{split}.csv"
        out.to_csv(csv_path, index=False)
        print(f"    -> {csv_path}  ({len(out):,} แถว)")

    # --- ตรวจสอบหลังสร้างเสร็จ (fail fast ถ้ามีอะไรผิด) ---
    print("\nตรวจสอบผลลัพธ์:")
    tr = pd.read_csv(args.split_dir / f"{new_variant}_train.csv", dtype={"patient_id": str})
    va = pd.read_csv(args.split_dir / f"{new_variant}_val.csv", dtype={"patient_id": str})
    te = pd.read_csv(args.split_dir / f"{new_variant}_test.csv", dtype={"patient_id": str})

    assert va["aug_index"].max() == 0, "ชุด val มีสำเนาที่ถูก augment ปนอยู่"
    assert te["aug_index"].max() == 0, "ชุด test มีสำเนาที่ถูก augment ปนอยู่"
    print(f"  [ok] val/test มีแต่สำเนาที่ 0 (ไม่ถูก augment) — {len(va):,} / {len(te):,} แถว")

    leak = set(tr["patient_id"]) & (set(va["patient_id"]) | set(te["patient_id"]))
    assert not leak, f"คนไข้รั่วข้าม split: {sorted(leak)[:5]}"
    print(f"  [ok] ไม่มีคนไข้ซ้ำระหว่าง train กับ val/test")

    assert tr["crop_path"].nunique() == len(tr), "มี path ซ้ำในชุด train"
    print(f"  [ok] path ไม่ซ้ำกัน — train {len(tr):,} แถว "
          f"({len(tr) / len(dfs['train']):.1f}x ของเดิม)")

    # --- เขียน config ตัวอย่างให้พร้อมรัน ---
    cfg_dir = Path("configs")
    cfg_dir.mkdir(exist_ok=True)

    # ระบุ --epochs มา = เขียน config ตัวเดียวตามที่สั่ง (ไม่เดาให้)
    # ไม่ระบุ = เขียน 2 ตัว: ตัวที่เทียบกับตาราง 15 epoch เดิมได้อย่างแฟร์ และตัวที่รันยาว
    if args.epochs is not None:
        plans = [(args.epochs, args.patience, f"{args.epochs}ep{args.patience}p")]
    else:
        plans = [(epoch_equiv, 3, "fair"), (15, 3, "long")]

    for epochs, patience, suffix in plans:
        name = f"effb0_{new_variant.replace('xray_masked_', 'masked_')}_{suffix}"
        cfg_path = cfg_dir / f"stage2_{name}.yaml"
        cfg_path.write_text(CONFIG_TEMPLATE.format(
            name=name, variant=new_variant, epochs=epochs, patience=patience,
            n_train=len(tr), n_orig=len(dfs["train"]),
            factor=round(expansion, 2), equiv=round(epochs * expansion),
        ), encoding="utf-8")
        print(f"  [ok] เขียน config: {cfg_path}")

    # พิมพ์คำสั่งจาก config ที่เขียนไปจริง ไม่ใช่ชื่อที่เดาเอา — ถ้าเดาแล้วชื่อไม่ตรง
    # (เช่นตอนระบุ --epochs มาเอง) คนที่ copy ไปรันจะเจอ error หาไฟล์ไม่เจอ
    print(f"\nเสร็จ — รันต่อได้เลยด้วย:")
    for _, _, suffix in plans:
        name = f"effb0_{new_variant.replace('xray_masked_', 'masked_')}_{suffix}"
        print(f"  python scripts/train.py --config configs/stage2_{name}.yaml")


if __name__ == "__main__":
    main()
