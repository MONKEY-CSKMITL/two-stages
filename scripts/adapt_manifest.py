"""
adapt_manifest.py — เชื่อมผลลัพธ์จาก crop.py เข้ากับ grade จาก Excel ให้เป็น manifest ที่พร้อมเทรน
 
USAGE:
    python ./scripts/adapt_manifest.py --crops_dir data/interim/crops --label_xlsx data/raw/DataTable.xlsx --id_width 4 --out_dir data/processed

input:  data/interim/crops/manifest.csv
            คอลัมน์: image, label, x0, y0, x1, y1, w, h, pixels
            (ใช้จริงแค่ 2 คอลัมน์: image = รหัสคนไข้, label = level 1-15)
            ที่เหลือเป็น log ไว้ตรวจสอบย้อนหลัง ไม่ได้อ่านในไฟล์นี้
 
        data/interim/crops/{patient_id}/*.png
            ไฟล์รูปที่ crop.py ตัดไว้ ทั้งแบบ _xray_bbox และ _xray_masked
            (เปิดดูแค่ว่ามีไฟล์อยู่จริงไหม ไม่ได้เปิดอ่านเนื้อในรูป)
 
        data/raw/DataTable.xlsx
            sheet "Main" คอลัมน์ No (รหัสคนไข้) + fx_T3..fx_L5 (grade ของแต่ละปล้อง)
            นี่คือแหล่ง grade จริงเพียงแหล่งเดียว
 
output: data/processed/manifest_xray_bbox.csv
        data/processed/manifest_xray_masked.csv
            คอลัมน์: patient_id, level_index, level_name, crop_path, grade_raw
                เช่น  0002, 2, T4, data/interim/crops/0002/0002_L02_T4_g0_xray_bbox.png, 0
                ได้ 2 ไฟล์ (1 ไฟล์ต่อ 1 แบบ crop) เอาไว้เทรนเทียบกันทีหลัง
 
หมายเหตุ: `label` ใน manifest.csv คือ level (1..15 = T3..L5) ไม่ใช่ grade —
grade ในชื่อไฟล์เป็นแค่ป้ายไว้เช็คด้วยตา ไม่ได้ถูกอ่านาใช้ในนี้
grade จริงมาจาก DataTable.xlsx เท่านั้น
"""

# --- import library ที่ต้องใช้ ---
import argparse              # อ่าน argument จาก command line เช่น --crops_dir
from pathlib import Path     # จัดการ path แบบข้ามระบบปฏิบัติการ

import pandas as pd          # อ่าน/เขียนตาราง (CSV, Excel)


# รายชื่อคอลัมน์ grade ใน Excel เรียงจาก T3 (บนสุด) ถึง L5 (ล่างสุด)
# ตำแหน่งในลิสต์นี้ตรงกับ level_index - 1 เสมอ (index 0 = fx_T3 = level 1)
FX_COLS = [
    "fx_T3", "fx_T4", "fx_T5", "fx_T6", "fx_T7", "fx_T8", "fx_T9",
    "fx_T10", "fx_T11", "fx_T12", "fx_L1", "fx_L2", "fx_L3", "fx_L4", "fx_L5",
]
# ตัดคำว่า "fx_" ออกจากแต่ละชื่อ เหลือแค่ชื่อกระดูกล้วนๆ เช่น "fx_T3" -> "T3"
# c[3:] หมายถึง "ตัดตัวอักษร 3 ตัวแรกทิ้ง" (f, x, _ = 3 ตัว)
LEVEL_NAMES = [c[3:] for c in FX_COLS]

# รูปแบบไฟล์ที่ crop.py ผลิตไว้ 2 แบบ ต้องประมวลผลทั้งคู่
VARIANTS = ["xray_bbox", "xray_masked"]


def load_label_table(xlsx_path, id_width):
    """
    อ่าน DataTable.xlsx ทั้งไฟล์ครั้งเดียว แปลงเป็น dict (ตารางค้นหาแบบเร็ว)
    ผลลัพธ์: {"0002": ["0","0","1",...15 ค่า...], "0003": [...], ...}
    (เหมือนฟังก์ชัน load_grade_lookup ใน crop.py แต่ชื่อฟังก์ชันต่างกันเฉยๆ
    ทำหน้าที่เดียวกัน: อ่าน Excel มาเก็บเป็น dict ไว้ค้นหาเร็วๆ ทีหลัง)
    """
    df = pd.read_excel(xlsx_path, sheet_name="Main")   # เปิด Excel, sheet ชื่อ "Main"
    label_map = {}                                        # dict ว่างไว้เก็บผลลัพธ์
    for _, row in df.iterrows():                          # วนทีละแถวของตาราง Excel (แถว = 1 คนไข้)
        # row["No"] คือเลขคนไข้ -> แปลงเป็น string เติม 0 ข้างหน้าให้ยาวเท่า id_width
        pid = str(int(row["No"])).zfill(id_width)
        # ดึงค่าทุกคอลัมน์ grade (fx_T3..fx_L5) ของแถวนี้ เก็บเป็น list ของ string
        label_map[pid] = [str(row[c]) for c in FX_COLS]
    return label_map   # คืน dict ทั้งหมดกลับไป


def find_crop_file(patient_dir: Path, pid: str, level: int, variant: str) -> Path | None:
    """
    หาไฟล์ crop ด้วย pattern (glob) แทนการต่อชื่อไฟล์แบบเดารูปแบบตรงๆ
    ทนต่อการเปลี่ยนชื่อไฟล์ในอนาคต เช่น crop.py เปลี่ยนจาก "{pid}_L02_xray_bbox.png"
    เป็น "{pid}_L02_T4_g0_xray_bbox.png" ก็ยังหาเจอ เพราะ pattern สนใจแค่
    "ขึ้นต้นด้วย L02 ลงท้ายด้วย _xray_bbox.png" ไม่สนใจว่าตรงกลางมีอะไรแทรก
    """
    # เครื่องหมาย * ใน pattern แปลว่า "อะไรก็ได้ กี่ตัวอักษรก็ได้" (wildcard)
    # เช่น pattern "0002_L02_*_xray_bbox.png" จะจับคู่กับ "0002_L02_T4_g0_xray_bbox.png" ได้
    pattern = f"{pid}_L{level:02d}_*_{variant}.png"

    # patient_dir.glob(pattern) = ค้นหาไฟล์ในโฟลเดอร์นี้ที่ชื่อตรงกับ pattern
    # sorted() เรียงผลลัพธ์ให้แน่นอน เผื่อมีมากกว่า 1 ไฟล์ตรง pattern
    matches = sorted(patient_dir.glob(pattern))

    if not matches:          # ถ้าลิสต์ผลลัพธ์ว่างเปล่า (หาไฟล์ไม่เจอเลย)
        return None            # คืนค่า None บอกว่าไม่เจอ

    if len(matches) > 1:     # ถ้าเจอมากกว่า 1 ไฟล์ตรง pattern (ผิดปกติ ไม่ควรเกิด)
        print(f"WARNING: เจอไฟล์ตรง pattern {pattern} มากกว่า 1 ไฟล์ ใช้ไฟล์แรก: {matches[0].name}")

    return matches[0]        # คืนไฟล์แรกที่เจอ (ปกติจะมีแค่ไฟล์เดียวตรง pattern)


def main():
    # --- ตั้งค่า argument ที่รับจาก command line ---
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops_dir", required=True, type=Path, help="โฟลเดอร์ crops/ ที่ crop.py สร้างไว้ (มี manifest.csv + โฟลเดอร์ย่อยต่อคนไข้)")
    ap.add_argument("--label_xlsx", required=True, type=Path)          # path ไฟล์ Excel (บังคับใส่)
    ap.add_argument("--id_width", type=int, default=4)                  # ความยาวรหัสคนไข้ (default 4 หลัก)
    ap.add_argument("--out_dir", type=Path, default=None,
                    help="ที่จะเขียน manifest ที่ join grade แล้ว (default: crops_dir)")
    args = ap.parse_args()   # อ่านค่าจริงที่ผู้ใช้พิมพ์ตอนรันคำสั่ง

    # ถ้าผู้ใช้ไม่ระบุ --out_dir มา ใช้ crops_dir แทน (เขียนทับที่เดียวกับ input)
    out_dir = args.out_dir or args.crops_dir
    out_dir.mkdir(parents=True, exist_ok=True)   # สร้างโฟลเดอร์ output ถ้ายังไม่มี

    # เปิด manifest.csv ที่ crop.py เขียนไว้ (มีแค่ level ไม่มี grade)
    # dtype={...} บังคับให้อ่านคอลัมน์ "image" เป็น string (กันเลข 0002 โดน pandas ตัด 0 นำหน้าทิ้ง)
    # และคอลัมน์ "label" เป็น int (ตัวเลขล้วนๆ เอาไว้คำนวณ)
    crops_manifest = pd.read_csv(args.crops_dir / "manifest.csv",
                                 dtype={"image": str, "label": int})

    # โหลดตาราง grade จาก Excel เตรียมไว้ (เรียกฟังก์ชันด้านบน)
    labels = load_label_table(args.label_xlsx, args.id_width)

    # เตรียมที่เก็บผลลัพธ์แยกตาม variant (bbox / masked) แต่ละอันเป็น list ว่างเริ่มต้น
    # {v: [] for v in VARIANTS} คือ dict comprehension สร้าง {"xray_bbox": [], "xray_masked": []}
    per_variant = {v: [] for v in VARIANTS}
    missing_grade = 0                              # ตัวนับ: กี่แถวที่หา patient ใน Excel ไม่เจอ
    missing_file = {v: 0 for v in VARIANTS}         # ตัวนับ: กี่แถวที่หาไฟล์รูปไม่เจอ (แยกตาม variant)

    # วนทุกแถวใน manifest.csv (1 แถว = 1 ปล้องที่ crop.py ตัดสำเร็จมาแล้ว)
    for _, r in crops_manifest.iterrows():
        pid = str(r["image"]).zfill(args.id_width)   # รหัสคนไข้ของแถวนี้ (เติม 0 ให้ครบหลัก)
        level = int(r["label"])                        # level ของแถวนี้ (1-15)

        if not (1 <= level <= 15):   # ถ้า level ผิดปกตินอกช่วง (ข้อมูลเสีย)
            continue                   # ข้ามแถวนี้ไปเลย

        level_name = LEVEL_NAMES[level - 1]   # แปลง level ตัวเลขเป็นชื่อกระดูก เช่น 4 -> "T6"

        grade_raw = ""              # ค่าเริ่มต้น (ว่างเปล่า) เผื่อหา patient ใน Excel ไม่เจอ
        if pid in labels:            # ถ้ารหัสคนไข้นี้มีอยู่ใน Excel
            grade_raw = labels[pid][level - 1]   # ดึง grade ของ level นี้ออกมา (ลบ 1 เพราะ list เริ่มนับจาก 0)
        else:                        # ถ้าไม่มีคนไข้นี้ใน Excel เลย
            missing_grade += 1        # นับไว้เป็นสถิติ (จะพิมพ์เตือนตอนท้าย)

        patient_dir = args.crops_dir / pid   # path โฟลเดอร์ของคนไข้คนนี้ เช่น crops/0002/

        # วนทั้ง 2 variant (bbox, masked) หาไฟล์ที่ตรงแต่ละแบบ
        for variant in VARIANTS:
            crop_path = find_crop_file(patient_dir, pid, level, variant)   # ค้นหาไฟล์จริงด้วย pattern

            if crop_path is None:          # ถ้าหาไฟล์ไม่เจอ (อาจ crop.py ยังไม่ได้ตัดปล้องนี้)
                missing_file[variant] += 1   # นับไว้เป็นสถิติ
                continue                      # ข้าม variant นี้ของแถวนี้ไป

            # เจอไฟล์แล้ว -> เพิ่มแถวใหม่เข้า list ของ variant นี้
            per_variant[variant].append({
                "patient_id": pid,
                "level_index": level,
                "level_name": level_name,
                "crop_path": str(crop_path),   # แปลง Path object เป็น string ธรรมดา (เก็บลง CSV ได้)
                "grade_raw": grade_raw,
            })

    # เขียนผลลัพธ์แต่ละ variant เป็นไฟล์ CSV แยกกัน (manifest_xray_bbox.csv, manifest_xray_masked.csv)
    for vname, rows in per_variant.items():        # วนทีละ variant พร้อม list ของแถวที่สะสมไว้
        out_csv = out_dir / f"manifest_{vname}.csv"   # ตั้งชื่อไฟล์ output ตาม variant
        pd.DataFrame(rows).to_csv(out_csv, index=False)   # แปลง list of dict เป็นตาราง แล้วเซฟเป็น CSV
        print(f"{vname:12s}: {len(rows):6d} rows -> {out_csv}"
              f"   (missing files: {missing_file[vname]})")   # รายงานสรุปให้เห็นทันที

    # ถ้ามีแถวที่หา patient ใน Excel ไม่เจอเลยสักคน ให้เตือนไว้ท้ายสุด
    if missing_grade:
        print(f"\nWARNING: {missing_grade} crop-rows had a patient_id not in the label xlsx")


# ถ้ารันไฟล์นี้ตรงๆ (python adapt_manifest.py) ให้เรียกฟังก์ชัน main()
# ถ้าไฟล์นี้ถูก import ไปใช้ในไฟล์อื่น จะไม่รัน main() อัตโนมัติ
if __name__ == "__main__":
    main()