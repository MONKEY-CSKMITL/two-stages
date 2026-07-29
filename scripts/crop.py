"""
scripts/crop.py — ตัดกระดูกสันหลังแต่ละปล้องจาก mask + X-ray ต้นฉบับ

USAGE:
    python ./scripts/crop.py

input:  data/raw/masks/{patient_id}_mask.png
        data/raw/hologic/{patient_id}.png
        data/raw/DataTable.xlsx

output: data/interim/crops/manifest.csv
        data/interim/crops/{patient_id}/{patient_id}_L{level:02d}_{ชื่อกระดูก}_g{grade}_xray_bbox.png
        data/interim/crops/{patient_id}/{patient_id}_L{level:02d}_{ชื่อกระดูก}_g{grade}_xray_masked.png
        เช่น 002_L02_T4_g0_xray_bbox.png     (มีพื้นหลังติดมาด้วย)
            0002_L02_T4_g0_xray_masked.png  (ลบพื้นหลังออก เหลือแต่กระดูกปล้องนั้น)
"""

# --- import library ที่ต้องใช้ ---
import csv                   # เขียนไฟล์ manifest.csv (ตารางบัญชี)
import glob                  # ค้นหาไฟล์ตามรูปแบบชื่อ เช่น "*_mask.png"
import os                    # จัดการชื่อไฟล์/path ทั่วไป
from pathlib import Path     # จัดการ path แบบใหม่ ใช้ได้ทั้ง Windows/Mac/Linux โดยไม่ต้องกังวลเรื่อง \ กับ /

import numpy as np           # จัดการ "ตารางตัวเลข" ของรูปภาพ (mask, X-ray) อย่างรวดเร็ว
import pandas as pd          # อ่านไฟล์ Excel (DataTable.xlsx) เป็นตาราง
from PIL import Image        # เปิด/บันทึกไฟล์รูปภาพ (.png)


# --- กำหนด path ทั้งหมดไว้ล่วงหน้า (แก้ตรงนี้ที่เดียว ถ้าย้ายโฟลเดอร์) ---
# __file__ คือตำแหน่งของไฟล์โค้ดนี้เอง (scripts/crop.py)
# .parent ครั้งแรก = ขึ้นไปที่โฟลเดอร์ scripts/
# .parent ครั้งที่สอง = ขึ้นไปอีกชั้นถึง two_stage_project/ (root ของโปรเจกต์)
ROOT = Path(__file__).resolve().parent.parent

MASK_DIR = ROOT / "data" / "raw" / "masks"       # โฟลเดอร์เก็บไฟล์ mask (ตารางตัวเลขบอกตำแหน่งปล้อง)
XRAY_DIR = ROOT / "data" / "raw" / "hologic"     # โฟลเดอร์เก็บภาพ X-ray ต้นฉบับ (เนื้อกระดูกจริง)
XLSX_PATH = ROOT / "data" / "raw" / "DataTable.xlsx"  # ไฟล์ Excel ที่มี grade ของทุกปล้อง
OUT_DIR = ROOT / "data" / "interim" / "crops"    # ที่จะเซฟรูป crop + manifest.csv ที่ได้
ID_WIDTH = 4   # ความยาวรหัสคนไข้ เช่น เลข 2 -> เติม 0 ข้างหน้าจนยาว 4 ตัว -> "0002"

# รายชื่อปล้องกระดูก เรียงจากบนสุด(T3)ลงล่างสุด(L5) ตามลำดับกายวิภาคจริงของร่างกาย
# ตำแหน่งที่ 0 ในลิสต์นี้ (index 0) คือ "T3" ซึ่งตรงกับ level_index=1 (ต้องลบ 1 ตอนใช้งาน)
LEVEL_NAMES = ["T3", "T4", "T5", "T6", "T7", "T8", "T9", "T10", "T11", "T12",
               "L1", "L2", "L3", "L4", "L5"]

# สร้างชื่อคอลัมน์ Excel ที่จะไปอ่าน เช่น "fx_T3", "fx_T4", ... "fx_L5"
# (list comprehension: สร้าง list ใหม่โดยเติมคำว่า "fx_" หน้าทุกชื่อใน LEVEL_NAMES)
FX_COLS = [f"fx_{lv}" for lv in LEVEL_NAMES]


def load_grade_lookup(xlsx_path: Path, id_width: int) -> dict:
    """
    อ่าน DataTable.xlsx ทั้งไฟล์ครั้งเดียว แปลงเป็น dict (ตารางค้นหาแบบเร็ว)
    เพื่อเอา grade ไปติดชื่อไฟล์ตอน crop (แค่ไว้ดูตา ไม่ใช่แหล่งความจริงหลัก)
    ผลลัพธ์: {"0002": ["0","0","1",...15 ค่า...], "0003": [...], ...}
    """
    xl = pd.read_excel(xlsx_path, sheet_name="Main")   # เปิด Excel, sheet ชื่อ "Main"
    lookup = {}                                          # dict ว่างไว้เก็บผลลัพธ์
    for _, r in xl.iterrows():                           # วนทีละแถวของตาราง Excel (แต่ละแถว = 1 คนไข้)
        # r["No"] คือเลขคนไข้ (อาจเป็น 2, 3, ...) -> แปลงเป็น string แล้วเติม 0 ข้างหน้า
        # เช่น 2 -> "0002" เพื่อให้ตรงกับชื่อไฟล์ mask/xray
        pid = str(int(r["No"])).zfill(id_width)
        # ดึงค่า grade จากคอลัมน์ fx_T3..fx_L5 ของแถวนี้ ทำเป็น list ของ string
        # (list comprehension อีกแบบ: ดึงค่า r[c] ของทุกชื่อคอลัมน์ c ใน FX_COLS)
        lookup[pid] = [str(r[c]) for c in FX_COLS]
    return lookup   # คืนตารางค้นหาทั้งหมดกลับไป


def main():
    # สร้างโฟลเดอร์ output ถ้ายังไม่มี (parents=True = สร้างทุกชั้นที่ขาด, exist_ok=True = ถ้ามีแล้วไม่ error)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # โหลดตาราง grade จาก Excel เตรียมไว้ (เรียกฟังก์ชันด้านบน)
    grade_lookup = load_grade_lookup(XLSX_PATH, ID_WIDTH)

    # ค้นหาไฟล์ mask ทุกไฟล์ในโฟลเดอร์ ที่ชื่อลงท้ายด้วย "_mask.png"
    # sorted() เรียงชื่อไฟล์ตามตัวอักษร กันไม่ให้ลำดับสุ่มไปมาทุกครั้งที่รัน
    mask_files = sorted(glob.glob(str(MASK_DIR / "*_mask.png")))

    if not mask_files:   # ถ้าลิสต์ว่างเปล่า (หาไฟล์ mask ไม่เจอเลย)
        raise FileNotFoundError(f"ไม่เจอไฟล์ mask ใน {MASK_DIR}")   # หยุดโปรแกรมพร้อมข้อความ error ที่ชัดเจน

    manifest_rows = []   # ลิสต์เก็บข้อมูลทุกแถวที่จะเขียนลง manifest.csv ตอนท้าย
    n_crops = 0           # ตัวนับ: ตัดไปแล้วกี่ปล้อง
    n_skipped = 0         # ตัวนับ: ข้ามไปกี่คนไข้ (เพราะหา X-ray คู่กันไม่เจอ หรือขนาดไม่ตรง)

    # วนทุกไฟล์ mask ที่เจอ (1 รอบ = 1 คนไข้)
    for mf in mask_files:
        # os.path.basename(mf) = ตัด path ออก เหลือแค่ชื่อไฟล์ เช่น "0002_mask.png"
        # .replace("_mask.png", "") = ตัดคำท้ายออก เหลือแค่รหัสคนไข้ "0002"
        stem = os.path.basename(mf).replace("_mask.png", "")

        # สร้าง path ของภาพ X-ray ที่ควรจะคู่กับ mask นี้ เช่น data/raw/hologic/0002.png
        xray_path = XRAY_DIR / f"{stem}.png"

        if not xray_path.exists():   # ถ้าไม่มีไฟล์ X-ray นี้อยู่จริง
            n_skipped += 1            # นับว่าข้าม 1 คน
            continue                  # ข้ามไปทำคนไข้คนถัดไปในลูป ไม่ทำโค้ดที่เหลือด้านล่าง

        # เปิดไฟล์ mask แล้วแปลงเป็น "ตารางตัวเลข" (numpy array)
        mask = np.array(Image.open(mf))
        if mask.ndim == 3:            # ถ้า mask ถูกเซฟมาเป็นภาพสี (มี 3 มิติ: สูง,กว้าง,ช่องสี)
            mask = mask[..., 0]       # เอาแค่ช่องแรก (เพราะ 3 ช่องมีค่าเหมือนกันหมดอยู่แล้ว)

        # เปิดภาพ X-ray, .convert("L") = บังคับให้เป็นภาพขาวดำช่องเดียว แล้วแปลงเป็นตารางตัวเลข
        xray = np.array(Image.open(xray_path).convert("L"))

        if xray.shape != mask.shape:   # ถ้าขนาดตาราง (แถว x คอลัมน์) ของสองไฟล์ไม่เท่ากัน
            n_skipped += 1             # แปลว่าพิกัดจาก mask จะเอาไปตัด X-ray ผิดตำแหน่งแน่นอน -> ข้าม
            continue

        # สร้างโฟลเดอร์ย่อยสำหรับคนไข้คนนี้ เช่น data/interim/crops/0002/
        img_out = OUT_DIR / stem
        img_out.mkdir(parents=True, exist_ok=True)

        # ดึงตาราง grade ของคนไข้คนนี้จากที่โหลดไว้ตอนต้น
        # .get(stem) คืนค่า None ถ้าหา stem ไม่เจอใน dict (กันโปรแกรม error ถ้าคนไข้ไม่มีใน Excel)
        patient_grades = grade_lookup.get(stem)

        # np.unique(mask) = หาตัวเลขทั้งหมดที่ปรากฏในตาราง mask แบบไม่ซ้ำกัน (เรียงจากน้อยไปมาก)
        # if v != 0 = กรองเอาค่า 0 (พื้นหลัง) ออก เพราะ 0 ไม่ใช่กระดูก
        # int(v) = แปลงจาก numpy type เป็น int ธรรมดาของ Python
        labels = [int(v) for v in np.unique(mask) if v != 0]

        # วนทุก level ที่เจอใน mask ของคนไข้คนนี้ (1 รอบ = 1 ปล้อง)
        for lab in labels:
            if not (1 <= lab <= 15):   # ถ้าค่าที่เจอไม่อยู่ในช่วง 1-15 (ผิดปกติ/ข้อมูลเสีย)
                continue                 # ข้ามปล้องนี้ไปเลย ไม่ตัดรูป

            # --- ขั้นตอนหา bounding box (กรอบสี่เหลี่ยม) ---
            sel = mask == lab            # สร้างตารางบูลีน: True เฉพาะช่องที่ค่า = lab (ปล้องที่กำลังดู)
            ys, xs = np.where(sel)       # แปลงตำแหน่ง True ทั้งหมดเป็นพิกัด (ys=แถวทั้งหมด, xs=คอลัมน์ทั้งหมด)
            y0, y1 = ys.min(), ys.max() + 1   # แถวบนสุด, แถวล่างสุด+1 (ขอบบน-ล่างของกรอบ)
            x0, x1 = xs.min(), xs.max() + 1   # คอลัมน์ซ้ายสุด, คอลัมน์ขวาสุด+1 (ขอบซ้าย-ขวาของกรอบ)
            # (+1 เพราะการตัด array ใน Python ไม่รวมค่าตัวสุดท้าย ต้องบวกเพิ่มให้กินขอบครบพอดี)

            # ตัดภาพ X-ray จริง (ไม่ใช่ mask) ตามกรอบพิกัดที่คำนวณได้ด้านบน
            # แบบที่ 1: bbox = เอาทุกอย่างในกรอบ รวมพื้นหลังและขอบปล้องข้างเคียงที่ติดมาด้วย
            xr = xray[y0:y1, x0:x1]

            # แบบที่ 2: masked = เอากรอบเดิม แต่ลบทุกอย่างที่ไม่ใช่กระดูกปล้องนี้ออก
            # sub คือตารางบูลีนของกรอบนี้ (True = เป็นกระดูกปล้องนี้จริง)
            sub = sel[y0:y1, x0:x1]
            # np.where(เงื่อนไข, ค่าถ้าจริง, ค่าถ้าเท็จ)
            #   ช่องไหน sub เป็น True  -> ใช้ค่าความสว่างจริงจาก xr (เห็นเนื้อกระดูก)
            #   ช่องไหน sub เป็น False -> ใส่ 0 (ดำสนิท = ลบพื้นหลัง/ปล้องข้างเคียงทิ้ง)
            xr_masked = np.where(sub, xr, 0).astype(np.uint8)

            # ดึง grade มาติดชื่อไฟล์ (ไว้ดูตาเฉยๆ)
            # patient_grades[lab - 1] เพราะ list เริ่มนับจาก index 0 แต่ level เริ่มนับจาก 1
            # ถ้า patient_grades เป็น None (หาไม่เจอใน Excel) ให้ใช้คำว่า "NA" แทน
            grade_tag = patient_grades[lab - 1] if patient_grades else "NA"

            # ดึงชื่อกระดูกจริง (T3, T4, ... L5) จาก LEVEL_NAMES ที่ประกาศไว้ตอนต้นไฟล์
            # ใช้ lab - 1 เพราะ level เริ่มนับจาก 1 แต่ list เริ่มนับจาก index 0
            # เช่น lab=2 -> LEVEL_NAMES[1] -> "T4"
            level_name = LEVEL_NAMES[lab - 1]

            # ประกอบชื่อไฟล์: รหัสคนไข้ + level ตัวเลข + ชื่อกระดูก + grade
            base = f"{stem}_L{lab:02d}_{level_name}_g{grade_tag}"

            # เซฟทั้ง 2 แบบ ต่างกันแค่ส่วนท้ายชื่อไฟล์ (_xray_bbox / _xray_masked)
            # เอาไว้เทรนเทียบกันทีหลังว่าการมี/ไม่มีพื้นหลังส่งผลต่อความแม่นยำแค่ไหน
            Image.fromarray(xr).save(img_out / f"{base}_xray_bbox.png")
            Image.fromarray(xr_masked).save(img_out / f"{base}_xray_masked.png")

            # เก็บข้อมูลของ crop นี้ไว้เป็น 1 แถว จะเอาไปเขียนลง manifest.csv ตอนท้าย
            # หมายเหตุ: ไม่มี grade ในนี้ (ตั้งใจ) ให้ adapt_manifest.py เป็นคน join grade อย่างเป็นทางการทีหลัง
            manifest_rows.append([
                stem, lab,
                int(x0), int(y0), int(x1), int(y1),      # พิกัดกรอบ
                int(x1 - x0), int(y1 - y0),                 # ความกว้าง, ความสูงของกรอบ
                int(sel.sum()),                              # จำนวนพิกเซลที่เป็นกระดูกจริง (นับ True ทั้งหมดใน sel)
            ])
            n_crops += 1   # นับเพิ่ม 1 ปล้องที่ตัดสำเร็จ

    # เปิดไฟล์ manifest.csv เพื่อเขียน (newline="" กันบรรทัดว่างแปลกๆ บน Windows)
    with open(OUT_DIR / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)                                   # สร้างตัวเขียน CSV
        w.writerow(["image", "label", "x0", "y0", "x1", "y1", "w", "h", "pixels"])  # เขียนหัวตาราง (ชื่อคอลัมน์)
        w.writerows(manifest_rows)                           # เขียนข้อมูลทุกแถวที่เก็บสะสมไว้ทั้งหมดทีเดียว

    # พิมพ์สรุปผลให้เห็นทันทีว่ารันสำเร็จแค่ไหน
    print(f"จำนวน mask ที่ประมวลผล : {len(mask_files)}")
    print(f"ข้ามไป (ไม่มี/ขนาดไม่ตรงกับ xray) : {n_skipped}")
    print(f"จำนวน crop ทั้งหมด : {n_crops}")
    print(f"โฟลเดอร์ output : {OUT_DIR}")


# ถ้ารันไฟล์นี้ตรงๆ (python crop.py) ให้เรียกฟังก์ชัน main()
# ถ้าไฟล์นี้ถูก import ไปใช้ในไฟล์อื่น จะไม่รัน main() อัตโนมัติ (กันรันซ้ำโดยไม่ตั้งใจ)
if __name__ == "__main__":
    main()