"""
run_split.py — แบ่งข้อมูลเป็น train/val/test แล้วเซฟเป็นไฟล์ พร้อมตรวจสอบว่าไม่มีข้อมูลรั่ว

input:  data/processed/manifest_xray_bbox.csv
        data/processed/manifest_xray_masked.csv
            (ใส่ได้หลายไฟล์ — คือ manifest ที่ adapt_manifest.py สร้างไว้)
            คอลัมน์ที่ใช้: patient_id, grade_raw
            (คอลัมน์อื่นไม่ได้ใช้ตอนแบ่ง แต่จะถูกคัดลอกติดไปในไฟล์ output ด้วย)

output: data/processed/splits/{variant}_train.csv
        data/processed/splits/{variant}_val.csv
        data/processed/splits/{variant}_test.csv
            แต่ละไฟล์มีคอลัมน์เหมือน manifest ต้นทางทุกอย่าง แค่แบ่งแถวออกเป็น 3 กอง
            เช่น xray_bbox_train.csv, xray_bbox_val.csv, xray_bbox_test.csv

        data/processed/splits/split_summary.csv
            ตารางสรุปว่าแต่ละกองมีคนไข้/ปล้อง/grade อย่างละเท่าไหร่

จุดสำคัญ: แบ่ง "ครั้งเดียว" แล้วเอาผลไปใช้กับทุก variant
เพราะถ้าแบ่งแยกกันแต่ละไฟล์ คนไข้คนเดียวกันอาจตกไปคนละกองในแต่ละ variant
ทำให้เทียบผล bbox vs masked กันไม่แฟร์ (และเสี่ยงข้อมูลรั่วข้ามการทดลอง)

USAGE:
    python3 run_split.py \
        --manifests data/processed/manifest_xray_bbox.csv data/processed/manifest_xray_masked.csv \
        --out_dir data/processed/splits \
        --val_frac 0.15 --test_frac 0.15 --seed 42
"""

# --- import library ที่ต้องใช้ ---
import argparse              # อ่าน argument จาก command line
import sys                   # ใช้เพิ่ม path ให้ Python หา module ของเราเจอ
from pathlib import Path     # จัดการ path แบบข้ามระบบปฏิบัติการ

import pandas as pd          # จัดการตาราง

# บอก Python ว่าให้ไปหา module ในโฟลเดอร์ src/ ด้วย
# (เพราะ split.py อยู่ที่ src/core/stage2/split.py ไม่ได้อยู่โฟลเดอร์เดียวกับไฟล์นี้)
# Path(__file__).parent.parent = ขึ้นจาก scripts/ ไปที่ root ของโปรเจกต์
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# import ฟังก์ชันที่เขียนไว้ใน split.py
from core.stage2.split import (load_usable_manifest, make_single_split,
                               assert_no_leakage, summarize)


def variant_name(manifest_path: Path) -> str:
    """
    แปลงชื่อไฟล์ manifest เป็นชื่อ variant สั้นๆ ไว้ตั้งชื่อไฟล์ output

    เช่น "manifest_xray_bbox.csv" -> "xray_bbox"
    """
    stem = manifest_path.stem   # .stem = ชื่อไฟล์ไม่รวมนามสกุล เช่น "manifest_xray_bbox"

    # .startswith(...) = เช็คว่าขึ้นต้นด้วยคำนี้ไหม
    # ถ้าใช่ ให้ตัดคำว่า "manifest_" ออก (len("manifest_") = 9 ตัวอักษร)
    # ถ้าไม่ใช่ ใช้ชื่อไฟล์เดิมทั้งอัน
    return stem[len("manifest_"):] if stem.startswith("manifest_") else stem


def main():
    # --- ตั้งค่า argument ที่รับจาก command line ---
    ap = argparse.ArgumentParser()
    # nargs="+" = รับได้หลายค่า (อย่างน้อย 1 ค่า) เช่น --manifests a.csv b.csv
    ap.add_argument("--manifests", required=True, nargs="+", type=Path,
                    help="ไฟล์ manifest ที่ adapt_manifest.py สร้างไว้ (ใส่ได้หลายไฟล์)")
    ap.add_argument("--out_dir", required=True, type=Path,
                    help="โฟลเดอร์ที่จะเซฟไฟล์ split")
    ap.add_argument("--val_frac", type=float, default=0.15)   # สัดส่วน val (default 15%)
    ap.add_argument("--test_frac", type=float, default=0.15)  # สัดส่วน test (default 15%)
    ap.add_argument("--seed", type=int, default=42)            # เลขสุ่ม (ใส่เลขเดิม = ได้ผลแบ่งเดิม)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)   # สร้างโฟลเดอร์ output ถ้ายังไม่มี

    # --- ขั้นที่ 1: โหลดทุก manifest เข้ามาเก็บไว้ ---
    variants = {}   # dict เก็บ {ชื่อ variant: ตารางข้อมูล}
    for mpath in args.manifests:              # วนทีละไฟล์ที่ผู้ใช้ใส่มา
        name = variant_name(mpath)             # แปลงชื่อไฟล์เป็นชื่อ variant
        df = load_usable_manifest(str(mpath))  # อ่าน + กรองแถวที่ใช้ไม่ได้ออก (99, x)
        variants[name] = df                     # เก็บลง dict
        print(f"โหลด {name:14s}: {len(df):6d} ปล้องที่ใช้ได้, "
              f"{df['patient_id'].nunique()} คนไข้")

    names = list(variants)   # list ของชื่อ variant ทั้งหมด เช่น ["xray_bbox", "xray_masked"]

    # --- ตรวจสอบว่าทุก variant มีคนไข้ชุดเดียวกัน (ควรเป็นแบบนั้น) ---
    base_pids = set(variants[names[0]]["patient_id"])   # ชุดคนไข้ของไฟล์แรก ใช้เป็นตัวเทียบ
    for n in names[1:]:                                  # วนไฟล์ที่เหลือ (ข้ามไฟล์แรก)
        other_pids = set(variants[n]["patient_id"])
        if other_pids != base_pids:                      # ถ้าชุดคนไข้ไม่ตรงกัน
            # - (ลบ) ระหว่าง set = หาสมาชิกที่มีในชุดแรกแต่ไม่มีในชุดหลัง
            only_a = base_pids - other_pids
            only_b = other_pids - base_pids
            print(f"เตือน: คนไข้ใน {names[0]} กับ {n} ไม่ตรงกัน "
                  f"(มีแค่ใน {names[0]}: {len(only_a)} คน, มีแค่ใน {n}: {len(only_b)} คน)")

    # --- ขั้นที่ 2: แบ่งข้อมูล "ครั้งเดียว" โดยใช้ไฟล์แรกเป็นตัวตั้ง ---
    tr0, va0, te0 = make_single_split(variants[names[0]],
                                      val_frac=args.val_frac,
                                      test_frac=args.test_frac,
                                      seed=args.seed)

    # ดึงเอาแค่ "รายชื่อคนไข้" ของแต่ละกองออกมา (ทิ้งข้อมูลปล้องไป)
    # เพราะจะเอารายชื่อนี้ไปใช้กรอง variant อื่นให้ได้การแบ่งเหมือนกันเป๊ะ
    pid_sets = {
        "train": set(tr0["patient_id"]),
        "val": set(va0["patient_id"]),
        "test": set(te0["patient_id"]),
    }
    print(f"\nแบ่งครั้งเดียว -> train {len(pid_sets['train'])} คน / "
          f"val {len(pid_sets['val'])} คน / test {len(pid_sets['test'])} คน")

    # --- ขั้นที่ 3: เอารายชื่อคนไข้ไปกรองทุก variant แล้วเซฟเป็นไฟล์ ---
    summary_rows = []   # list เก็บข้อมูลสรุป จะเขียนเป็นตารางตอนท้าย

    for name, df in variants.items():       # วนทีละ variant
        # กรองตารางด้วยรายชื่อคนไข้แต่ละกอง (.isin = อยู่ในชุดนี้ไหม)
        tr = df[df["patient_id"].isin(pid_sets["train"])].reset_index(drop=True)
        va = df[df["patient_id"].isin(pid_sets["val"])].reset_index(drop=True)
        te = df[df["patient_id"].isin(pid_sets["test"])].reset_index(drop=True)

        # ตรวจสอบว่าไม่มีคนไข้ซ้ำข้ามกอง (ถ้าเจอจะ raise error หยุดโปรแกรมทันที)
        assert_no_leakage(tr, va, te)

        # เซฟทั้ง 3 กองเป็นไฟล์ CSV แยกกัน
        # zip(...) = จับคู่ 2 list เข้าด้วยกัน วนพร้อมกันทีละคู่
        for split_name, d in zip(["train", "val", "test"], [tr, va, te]):
            out_csv = args.out_dir / f"{name}_{split_name}.csv"
            d.to_csv(out_csv, index=False)   # index=False = ไม่ต้องเซฟเลขลำดับแถวลงไฟล์

            # เก็บข้อมูลสรุปของกองนี้
            # **summarize(d) = แตก dict ที่ได้จาก summarize ออกมารวมกับ dict นี้
            summary_rows.append({"variant": name, "split": split_name, **summarize(d)})

    # --- ขั้นที่ 4: เขียนตารางสรุป + พิมพ์ให้ดู ---
    summ = pd.DataFrame(summary_rows)
    summ.to_csv(args.out_dir / "split_summary.csv", index=False)

    print("\n=== สรุปจำนวนปล้องแยกตาม grade ===")
    print(summ.to_string(index=False))   # to_string = พิมพ์ตารางให้อ่านง่ายใน terminal
    print(f"\nเซฟไฟล์ทั้งหมดไว้ที่ -> {args.out_dir}")


# ถ้ารันไฟล์นี้ตรงๆ ให้เรียก main()
if __name__ == "__main__":
    main()