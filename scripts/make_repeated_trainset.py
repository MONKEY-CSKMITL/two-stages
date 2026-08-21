"""
make_repeated_trainset.py — ขยายชุด train ด้วยการ "ทำซ้ำแถว" ไม่ใช่ "สร้างไฟล์ใหม่"

--------------------------------------------------------------------------
ทำไมต้องมีตัวนี้ ทั้งที่มี build_augmented_trainset.py อยู่แล้ว
--------------------------------------------------------------------------
build_augmented_trainset.py อบ augment ลงไฟล์ ทำให้เกิดความไม่สมมาตรที่วัดได้:

    normal   สำเนาที่ 0 อย่างเดียว -> augment สุ่มสดตอนเทรน 1 ชั้น
    damaged  สำเนาที่ 1-9 (90%)    -> augment ที่อบไว้ + สุ่มสดอีก = 2 ชั้น

วัดจากปล้องผิดปกติจริง 40 ใบ: 1 ชั้นห่างจากภาพจริง 23.24 ระดับ ส่วน 2 ชั้น 40.75
= **บิดเบือนมากกว่า 1.75 เท่า**

ผลคือโมเดลเรียนว่า "damaged = ปล้องที่บิดเบี้ยวหนัก" แต่ชุด test ไม่ถูก augment เลย
ปล้อง mild จริงจึงไม่เหมือน damaged ที่โมเดลรู้จัก -> โมเดลทายว่า normal
(วัดได้จาก bin_x10: ปล่อย damaged หลุด 42 ใบ เทียบกับชุดเดิมที่ 30 ใบ)

ไฟล์นี้แก้ที่ต้นเหตุ: **ทุกแถวชี้ไฟล์ต้นฉบับเดียวกัน** ไม่มีอะไรอบไว้ล่วงหน้า
augment สุ่มสดตอนเทรนจึงทำงานกับทุกคลาสเท่ากัน 1 ชั้นเสมอ ความไม่สมมาตรหายไป
แต่ยังได้ประโยชน์ที่ตั้งใจ: คลาสน้อยถูกหยิบมาอัปเดต gradient N เท่าต่อ epoch
ด้วยหน้าตาที่ต่างกันทุกครั้ง

ผลพลอยได้: ไม่เขียนไฟล์ภาพใหม่เลยสักไฟล์ (ประหยัด ~190 MB และเวลาสร้าง ~10 นาที)

USAGE:
    python scripts/make_repeated_trainset.py --class_factors "0:1,1:10,2:10,3:10"
    python scripts/make_repeated_trainset.py --class_factors "0:1,1:10,2:10,3:10" --dry_run
"""

import argparse
import sys

import pandas as pd

sys.path.insert(0, str(__file__.rsplit("scripts", 1)[0] + "src"))

from core.stage2.dataset import load_split_csv

GRADE_NAMES = {0: "normal", 1: "mild", 2: "moderate", 3: "severe"}


def parse_factors(spec: str) -> dict:
    return {int(k): int(v) for k, v in (p.split(":") for p in spec.split(","))}


def main():
    # Windows พิมพ์ CRLF ทำให้ชื่อที่ shell รับไปมี CR ติดท้าย (เคยพังมาแล้ว)
    sys.stdout.reconfigure(newline=chr(10))

    ap = argparse.ArgumentParser()
    ap.add_argument("--split_dir", default="data/processed/splits")
    ap.add_argument("--variant", default="xray_masked")
    ap.add_argument("--class_factors", default="0:1,1:10,2:10,3:10")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    factors = parse_factors(args.class_factors)
    tag = "rep" + "-".join(str(factors.get(g, 1)) for g in [0, 1, 2, 3])
    new_variant = f"{args.variant}_{tag}"

    print(f"variant ใหม่: {new_variant}")
    print(f"  ตัวคูณรายคลาส: {factors}")
    print("  ทุกแถวชี้ไฟล์ต้นฉบับเดิม — ไม่สร้างไฟล์ภาพใหม่แม้แต่ไฟล์เดียว\n")

    for split in ["train", "val", "test"]:
        src = f"{args.split_dir}/{args.variant}_{split}.csv"
        df = load_split_csv(src, task="multiclass")

        if split == "train":
            # ทำซ้ำเฉพาะชุด train — val/test ต้องคงสัดส่วนธรรมชาติของโลกจริงเสมอ
            parts = []
            for g, n in factors.items():
                sub = df[df["grade_4class"] == g]
                for k in range(n):
                    s = sub.copy()
                    s["rep_index"] = k       # ไว้ตรวจย้อนหลังว่าแถวไหนเป็นสำเนาที่เท่าไหร่
                    parts.append(s)
            out = pd.concat(parts, ignore_index=True)
            # สลับแถวให้สำเนาของปล้องเดียวกันไม่มาติดกันเป็นก้อน — ถ้าเรียงติดกัน
            # batch หนึ่งอาจมีแต่สำเนาของปล้องเดียว ทำให้ gradient ของ batch นั้นเบ้
            out = out.sample(frac=1.0, random_state=0).reset_index(drop=True)
        else:
            out = df.copy()
            out["rep_index"] = 0

        n_before, n_after = len(df), len(out)
        if split == "train":
            print("  แผนชุด train:")
            print(f"    {'grade':<12}{'เดิม':>8}{'x':>4}{'ใหม่':>9}")
            for g in [0, 1, 2, 3]:
                nb = int((df["grade_4class"] == g).sum())
                print(f"    {GRADE_NAMES[g]:<12}{nb:>8,}{factors.get(g, 1):>4}{nb * factors.get(g, 1):>9,}")
            print(f"    {'รวม':<12}{n_before:>8,}{'':>4}{n_after:>9,}   ({n_after / n_before:.2f}x)")

        if not args.dry_run:
            dst = f"{args.split_dir}/{new_variant}_{split}.csv"
            out.to_csv(dst, index=False)
            print(f"  -> {dst}  ({n_after:,} แถว)")

    if args.dry_run:
        print("\n--dry_run: ไม่ได้เขียนไฟล์")
        return

    # --- ตรวจสอบหลังเขียน ---
    tr = pd.read_csv(f"{args.split_dir}/{new_variant}_train.csv", dtype={"patient_id": str})
    va = pd.read_csv(f"{args.split_dir}/{new_variant}_val.csv", dtype={"patient_id": str})
    te = pd.read_csv(f"{args.split_dir}/{new_variant}_test.csv", dtype={"patient_id": str})

    assert va["rep_index"].max() == 0 and te["rep_index"].max() == 0, "val/test ถูกทำซ้ำด้วย"
    print(f"\n  [ok] val/test ไม่ถูกทำซ้ำ — {len(va):,} / {len(te):,} แถว")

    leak = set(tr["patient_id"]) & (set(va["patient_id"]) | set(te["patient_id"]))
    assert not leak, f"คนไข้รั่วข้าม split: {sorted(leak)[:5]}"
    print("  [ok] ไม่มีคนไข้ซ้ำระหว่าง train กับ val/test")

    # ต่างจาก build_augmented_trainset.py ตรงนี้: path **ต้องซ้ำ** เพราะเป็นไฟล์เดียวกัน
    n_files = tr["crop_path"].nunique()
    print(f"  [ok] {len(tr):,} แถว ชี้ไปยังไฟล์จริง {n_files:,} ไฟล์ "
          f"(ซ้ำได้ตามตั้งใจ — ไม่มีไฟล์ใหม่ถูกสร้าง)")


if __name__ == "__main__":
    main()
