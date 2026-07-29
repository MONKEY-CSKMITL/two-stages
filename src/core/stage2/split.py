"""
split.py — ตรรกะการแบ่งข้อมูลเป็น train/val/test (ไฟล์นี้เป็น "ห้องสมุด" ไม่ได้รันเอง)

ไฟล์นี้ไม่มี main() ไม่ได้รันตรงๆ — เป็นชุดฟังก์ชันให้ scripts/run_split.py
และ scripts/train.py เรียกใช้ แยกออกมาเพราะตรรกะการแบ่งข้อมูลเป็นเรื่องของตาราง
ล้วนๆ ไม่เกี่ยวกับโมเดลเลย (ไม่ต้องใช้ torch ด้วยซ้ำ)

หลักการสำคัญ 2 ข้อที่ต้องยึดเสมอ:

  1. แบ่งที่ระดับ "คนไข้" ไม่ใช่ระดับ "ปล้อง"
     เพราะคนไข้ 1 คนมีหลายปล้อง ถ้าปล้องของคนเดียวกันไปอยู่คนละกอง
     โมเดลจะแอบเห็นกระดูกคนนั้นตอนเทรนแล้วไปเจออีกตอนสอบ = คะแนนสวยเกินจริง
     (เรียกปัญหานี้ว่า data leakage — ข้อมูลรั่วไหล)

  2. แบ่งแบบคุมสัดส่วน (stratify) ตาม "ชุด grade ที่คนไข้คนนั้นมี"
     เพราะคนไข้ที่มีปล้องหักรุนแรงมีน้อยมาก ถ้าสุ่มธรรมดาอาจกระจุกอยู่กองเดียว
     ทำให้อีกกองไม่มีตัวอย่างให้วัดผลเลย

ต้องติดตั้ง: pip install iterative-stratification
"""

# --- import library ที่ต้องใช้ ---
import pandas as pd   # จัดการตาราง (อ่าน CSV, กรองแถว, จัดกลุ่ม)

# MultilabelStratifiedKFold = อัลกอริทึมแบ่งข้อมูลแบบคุมสัดส่วนหลาย label พร้อมกัน
# (ต่างจาก stratify ธรรมดาของ sklearn ที่คุมได้ทีละ label เดียว)
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold


# --- แปลง grade ดิบจาก Excel เป็นตัวเลข 0-3 ที่โมเดลใช้ได้ ---
# "4" คือค่าพิมพ์ผิดใน Excel (Genant มีแค่ 0-3) ถือว่าเป็น 3 (รุนแรง)
LABEL_MAP = {"0": 0, "1": 1, "2": 2, "3": 3, "4": 3}

# ค่าที่ต้องตัดทิ้ง ใช้เทรนไม่ได้:
#   "99" = ปล้องอยู่นอกขอบเขตภาพ (มองไม่เห็น จึงไม่มีคำตอบที่ถูกต้อง)
#   "x"  = มีพยาธิสภาพอื่นที่ไม่ใช่การหัก (เช่น เนื้องอก, กระดูกเชื่อมติดกัน)
#   ที่เหลือคือค่าว่าง/ค่าผิดพลาดจากการอ่านไฟล์
DROP_VALUES = {"99", "x", "", "nan", "None"}


def load_usable_manifest(manifest_csv: str) -> pd.DataFrame:
    """
    อ่าน manifest ที่ adapt_manifest.py สร้างไว้ แล้วกรองเหลือเฉพาะแถวที่ใช้เทรนได้

    input:  path ของ manifest_xray_bbox.csv (หรือ _masked.csv)
    output: ตาราง pandas ที่กรองแล้ว + เพิ่มคอลัมน์ "label" (ตัวเลข 0-3)
    """
    # อ่านไฟล์ CSV เข้ามาเป็นตาราง
    # dtype={...} บังคับให้อ่าน 2 คอลัมน์นี้เป็น string
    #   - grade_raw: เพราะมีทั้งตัวเลขและตัวอักษร ("x") ปนกัน ถ้าปล่อยให้เดาเองจะพัง
    #   - patient_id: กัน pandas ตัดเลข 0 นำหน้าทิ้ง (0002 -> 2)
    df = pd.read_csv(manifest_csv, dtype={"grade_raw": str, "patient_id": str})

    # .fillna("") = เปลี่ยนช่องว่างเปล่า (NaN) เป็น string ว่าง กันโค้ดข้างล่างพัง
    # .str.strip() = ตัดช่องว่างหน้า-หลังออก เผื่อมีเว้นวรรคเกินมาจาก Excel
    df["grade_raw"] = df["grade_raw"].fillna("").str.strip()

    # สร้างเงื่อนไขกรอง: เก็บเฉพาะแถวที่ grade อยู่ใน LABEL_MAP และไม่อยู่ใน DROP_VALUES
    # .isin(...) = เช็คว่าค่าในคอลัมน์อยู่ในชุดที่กำหนดไหม (คืนค่า True/False ทุกแถว)
    # ~ (tilde) = กลับค่า True เป็น False (แปลว่า "ไม่อยู่ใน")
    # & = และ (ต้องเป็นจริงทั้ง 2 เงื่อนไข)
    keep = df["grade_raw"].isin(LABEL_MAP.keys()) & ~df["grade_raw"].isin(DROP_VALUES)

    # กรองเอาเฉพาะแถวที่ keep เป็น True, .copy() = สร้างตารางใหม่แยกจากของเดิม
    df = df[keep].copy()

    # .map(LABEL_MAP) = แปลงค่าทุกแถวตามตาราง เช่น "2" -> 2, "4" -> 3
    # .astype(int) = บังคับให้เป็นชนิดตัวเลขจำนวนเต็ม
    df["label"] = df["grade_raw"].map(LABEL_MAP).astype(int)

    # .reset_index(drop=True) = เรียงเลขลำดับแถวใหม่ 0,1,2,... (เพราะกรองแล้วเลขเดิมขาดหาย)
    return df.reset_index(drop=True)


def _patient_presence(df: pd.DataFrame) -> pd.DataFrame:
    """
    ยุบข้อมูลจาก "ระดับปล้อง" เป็น "ระดับคนไข้" เพื่อเตรียมแบ่งข้อมูล

    แต่ละคนไข้จะกลายเป็น 1 แถว พร้อมเวกเตอร์ 4 ช่องบอกว่า "มี grade อะไรบ้าง"
    เช่น คนไข้ที่มีทั้งปล้องปกติและปล้องหักปานกลาง -> [1, 0, 1, 0]
    (ไม่สนว่ามีอย่างละกี่ปล้อง สนแค่ "มี" หรือ "ไม่มี")

    ชื่อฟังก์ชันขึ้นต้นด้วย _ (underscore) เป็นธรรมเนียมบอกว่า
    "ฟังก์ชันนี้ใช้ภายในไฟล์นี้เท่านั้น ไม่ได้ตั้งใจให้ไฟล์อื่นเรียก"
    """
    # .groupby("patient_id")["label"] = จัดกลุ่มตามคนไข้ แล้วสนใจแค่คอลัมน์ label
    g = df.groupby("patient_id")["label"]

    # .apply(lambda s: set(s.unique())) = สำหรับแต่ละคนไข้ ให้หา grade ที่มีแบบไม่ซ้ำ
    # set(...) = ชุดของค่าที่ไม่ซ้ำกัน เช่น {0, 2} แปลว่ามีทั้งปกติและปานกลาง
    present = g.apply(lambda s: set(s.unique()))

    rows = []                              # list ว่างไว้เก็บผลลัพธ์
    for pid, grades in present.items():    # วนทีละคนไข้ (pid = รหัส, grades = set ของ grade)
        rows.append({
            "patient_id": pid,
            # int(0 in grades) = ถ้ามี grade 0 ให้เป็น 1 ถ้าไม่มีให้เป็น 0
            # (แปลง True/False เป็น 1/0 เพราะอัลกอริทึมต้องการตัวเลข)
            "has_0": int(0 in grades),
            "has_1": int(1 in grades),
            "has_2": int(2 in grades),
            "has_3": int(3 in grades),
        })
    return pd.DataFrame(rows)   # แปลง list of dict เป็นตาราง


def make_single_split(df: pd.DataFrame,
                      val_frac: float = 0.15,
                      test_frac: float = 0.15,
                      seed: int = 42):
    """
    แบ่งข้อมูลเป็น train/val/test ครั้งเดียว (ไม่ทำ K-fold เพราะเป็นแค่ PoC)

    input:  df = ตารางจาก load_usable_manifest()
            val_frac/test_frac = สัดส่วนที่ต้องการ (default 15% ต่อกอง เหลือ 70% เป็น train)
            seed = ตัวเลขกำหนดการสุ่ม (ใส่เลขเดิมได้ผลแบ่งเดิมทุกครั้ง = ทำซ้ำได้)
    output: (train_df, val_df, test_df) — 3 ตาราง ยังอยู่ในระดับปล้องเหมือนเดิม
    """
    # --- ขั้นที่ 1: แยก test ออกมาก่อน (ล็อกไว้ ไม่แตะจนจบงาน) ---
    pp = _patient_presence(df)                              # ยุบเป็นระดับคนไข้
    Y = pp[["has_0", "has_1", "has_2", "has_3"]].values      # ดึงเวกเตอร์ 4 ช่องออกมาเป็น array

    # n_splits = จำนวนกองที่จะแบ่ง คำนวณกลับจากสัดส่วนที่ต้องการ
    # เช่น test_frac=0.15 -> 1/0.15 = 6.67 -> ปัดเป็น 7 (แบ่ง 7 กอง เอา 1 กองเป็น test ≈ 14%)
    # max(2, ...) กันกรณีสัดส่วนใหญ่เกินจนคำนวณได้น้อยกว่า 2 (แบ่งน้อยกว่า 2 กองไม่ได้)
    n_splits_test = max(2, round(1 / test_frac))

    # สร้างตัวแบ่ง: shuffle=True = สลับลำดับก่อนแบ่ง, random_state=seed = ล็อกผลการสุ่ม
    mskf = MultilabelStratifiedKFold(n_splits=n_splits_test, shuffle=True, random_state=seed)

    # .split(...) คืน generator ของหลายรอบ (fold) แต่เราเอาแค่รอบแรก -> ใช้ next()
    # ได้ตำแหน่ง (index) ของคนไข้ 2 กลุ่ม: กลุ่มที่เหลือไว้ กับกลุ่มที่เป็น test
    trainval_pos, test_pos = next(mskf.split(pp["patient_id"].values, Y))

    # .iloc[ตำแหน่ง] = เลือกแถวตามตำแหน่งตัวเลข -> ได้รหัสคนไข้ของแต่ละกลุ่ม
    # set(...) = แปลงเป็นชุด ค้นหาเร็วกว่า list ตอนใช้ .isin() ข้างล่าง
    test_pids = set(pp.iloc[test_pos]["patient_id"])
    trainval_pids = set(pp.iloc[trainval_pos]["patient_id"])

    # กรองตารางเดิม (ระดับปล้อง) ให้เหลือเฉพาะคนไข้ในแต่ละกลุ่ม
    test_df = df[df["patient_id"].isin(test_pids)].reset_index(drop=True)
    trainval_df = df[df["patient_id"].isin(trainval_pids)].reset_index(drop=True)

    # --- ขั้นที่ 2: แยก val ออกจากส่วนที่เหลือ (คำนวณสัดส่วนใหม่) ---
    pp2 = _patient_presence(trainval_df)                      # ยุบเป็นระดับคนไข้อีกรอบ (เฉพาะกลุ่มที่เหลือ)
    Y2 = pp2[["has_0", "has_1", "has_2", "has_3"]].values

    # ต้องคำนวณสัดส่วนใหม่ เพราะตอนนี้ฐานเหลือแค่ 85% ไม่ใช่ 100% แล้ว
    # เช่น อยากได้ val 15% ของทั้งหมด แต่ตอนนี้เหลือ 85% -> ต้องเอา 0.15/0.85 = 17.6% ของที่เหลือ
    rel_val = val_frac / (1.0 - test_frac)
    n_splits_val = max(2, round(1 / rel_val))

    mskf2 = MultilabelStratifiedKFold(n_splits=n_splits_val, shuffle=True, random_state=seed)
    train_pos, val_pos = next(mskf2.split(pp2["patient_id"].values, Y2))

    train_pids = set(pp2.iloc[train_pos]["patient_id"])
    val_pids = set(pp2.iloc[val_pos]["patient_id"])

    train_df = trainval_df[trainval_df["patient_id"].isin(train_pids)].reset_index(drop=True)
    val_df = trainval_df[trainval_df["patient_id"].isin(val_pids)].reset_index(drop=True)

    return train_df, val_df, test_df


def assert_no_leakage(*dfs):
    """
    ตรวจสอบว่าไม่มีคนไข้คนไหนโผล่ในมากกว่า 1 กอง (ถ้าเจอ = ข้อมูลรั่ว ต้องหยุดทันที)

    *dfs (มีดอกจัน) = รับได้หลายตารางไม่จำกัดจำนวน เช่น assert_no_leakage(tr, va, te)
    """
    seen = {}                                  # dict เก็บว่า "คนไข้คนนี้เจอในกองที่เท่าไหร่แล้ว"
    for i, d in enumerate(dfs):                # enumerate = วนพร้อมนับเลขลำดับ (i = 0, 1, 2, ...)
        for pid in d["patient_id"].unique():   # วนทุกคนไข้ในกองนี้ (unique = ไม่ซ้ำ)
            if pid in seen:                    # ถ้าเคยเจอคนนี้ในกองก่อนหน้าแล้ว
                # raise = หยุดโปรแกรมทันทีพร้อมข้อความ error (ดีกว่าปล่อยให้เทรนต่อแล้วได้ผลผิด)
                raise AssertionError(f"คนไข้ {pid} อยู่ทั้งกอง {seen[pid]} และกอง {i} — ข้อมูลรั่ว!")
            seen[pid] = i                      # จดไว้ว่าเจอคนนี้ในกองที่ i แล้ว


def summarize(df: pd.DataFrame) -> dict:
    """
    สรุปว่ากองนี้มีคนไข้กี่คน กี่ปล้อง และแยกตาม grade ได้เท่าไหร่ (ไว้พิมพ์ดู)
    """
    # .value_counts() = นับจำนวนแต่ละค่า
    # .reindex(range(4)) = บังคับให้มีครบ 4 แถว (0,1,2,3) แม้บาง grade จะไม่มีเลย
    # .fillna(0) = ถ้า grade ไหนไม่มี ให้เป็น 0 แทนช่องว่าง
    dist = df["label"].value_counts().reindex(range(4)).fillna(0).astype(int)
    return {
        "patients": df["patient_id"].nunique(),   # nunique = นับจำนวนค่าที่ไม่ซ้ำ (= จำนวนคนไข้)
        "crops": len(df),                          # จำนวนแถวทั้งหมด (= จำนวนปล้อง)
        "normal": int(dist[0]),
        "mild": int(dist[1]),
        "moderate": int(dist[2]),
        "severe": int(dist[3]),
    }