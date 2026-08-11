"""
show_pipeline.py — ดูว่าภาพเดินผ่านอะไรบ้างก่อนถึงโมเดล (รันได้เลย ไม่ต้องเทรน)

input:  configs/xxx.yaml
            อ่านค่า data.preprocess, data.augment, data.resize_mode, model.backbone
            จาก config ตัวเดียวกับที่จะใช้เทรน — จึงมั่นใจได้ว่าที่เห็นคือที่จะได้จริง

output: outputs/pipeline/{ชื่อการทดลอง}/pipeline_stages.png
            ตารางภาพ แถว = ตัวอย่างแต่ละ grade, คอลัมน์ = ขั้นตอนในท่อ
        outputs/pipeline/{ชื่อการทดลอง}/pipeline_histograms.png
            การกระจายค่าความสว่างของพิกเซลกระดูก แยกตามขั้นตอน
        outputs/pipeline/{ชื่อการทดลอง}/pipeline_stats.csv
            ตัวเลขสรุปรายขั้นตอน (mean/sd/p1/p99/สัดส่วนที่อิ่มตัว)

ใช้ทำอะไร:
  ตรวจ config ใหม่ก่อนเสียเวลาเทรน — ถ้าตั้ง preprocess หรือ augment แรงเกินไป
  จะเห็นจากภาพทันทีภายในไม่กี่วินาที แทนที่จะรู้ตอนเทรนจบแล้วผลออกมาแย่

USAGE:
    python scripts/show_pipeline.py --config configs/stage2_effb0_masked_diag_aug_strong.yaml
    python scripts/show_pipeline.py --config ... --n_per_grade 2 --draws 4 --split train
"""

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from core.stage2.augment import get_augment_fn
from core.stage2.channels import get_channel_spec, warn_if_mask_unusable
from core.stage2.dataset import load_split_csv
from core.stage2.preprocessing import get_preprocess_fn
from core.utils.pipeline_viz import generate_pipeline_report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--split", default="train", choices=["train", "val", "test"],
                    help="ดูจากชุดไหน (ค่าเริ่มต้น train เพราะเป็นชุดเดียวที่โดน augment)")
    ap.add_argument("--n_per_grade", type=int, default=1, help="กี่ตัวอย่างต่อ 1 grade")
    ap.add_argument("--draws", type=int, default=3, help="สุ่ม augment กี่ครั้งให้ดู")
    ap.add_argument("--seed", type=int, default=0, help="เลขสุ่มสำหรับเลือกตัวอย่าง")
    ap.add_argument("--out_dir", type=Path, default=None,
                    help="ไม่ระบุ = outputs/pipeline/{ชื่อการทดลอง}")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    name = cfg["experiment"]["name"]
    split_dir = Path(cfg["data"]["split_dir"])
    variant = cfg["data"]["variant"]
    task = cfg["data"]["task"]
    backbone = cfg["model"]["backbone"]
    resize_mode = cfg["data"].get("resize_mode", "pad")

    preprocess_name = cfg["data"].get("preprocess") or "none"
    augment_name = cfg["data"].get("augment") or "none"
    channels_name = cfg["data"].get("channels") or "gray3"
    preprocess_fn = get_preprocess_fn(cfg["data"].get("preprocess"))
    channel_spec = get_channel_spec(cfg["data"].get("channels"))
    warn_if_mask_unusable(channel_spec, variant)

    # augment ทำเฉพาะชุด train — ถ้าขอดูชุดอื่น ต้องไม่แสดง augment เพื่อไม่ให้เข้าใจผิด
    # ว่าชุดวัดผลก็โดน augment ด้วย
    if args.split == "train":
        augment_fn = get_augment_fn(cfg["data"].get("augment"))
    else:
        augment_fn = None
        augment_name = f"{augment_name} (ไม่ใช้กับชุด {args.split})"

    df = load_split_csv(str(split_dir / f"{variant}_{args.split}.csv"), task=task)

    img_size = cfg["data"].get("img_size")
    size = img_size if img_size is not None else (518 if backbone == "rad_dino" else 224)

    out_dir = args.out_dir or Path("outputs/pipeline") / name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"การทดลอง : {name}")
    print(f"  backbone   = {backbone}   img_size = {size}   resize_mode = {resize_mode}")
    print(f"  preprocess = {preprocess_name}")
    print(f"  augment    = {augment_name}")
    print(f"  channels   = {channels_name} = {channel_spec}")
    print(f"  ชุดข้อมูล   = {args.split} ({len(df)} ปล้อง)")

    generate_pipeline_report(
        df, out_dir, prefix="pipeline",
        backbone=backbone, size=size, resize_mode=resize_mode,
        preprocess_fn=preprocess_fn, augment_fn=augment_fn, channel_spec=channel_spec,
        preprocess_name=preprocess_name, augment_name=augment_name,
        channels_name=channels_name,
        n_per_grade=args.n_per_grade, n_draws=args.draws, seed=args.seed,
    )

    print(f"\nเซฟไว้ที่ {out_dir}")
    for f in sorted(out_dir.glob("pipeline_*")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
