"""
dataset.py — โหลดข้อมูลจากไฟล์ split เข้าสู่รูปแบบที่โมเดลกินได้ (ไฟล์นี้เป็น "ห้องสมุด")

รองรับ 2 โจทย์ สลับได้ด้วยค่า task ตัวเดียว:

  task="multiclass"  ทำนาย Genant grade 4 ระดับ (0=ปกติ, 1=เล็กน้อย, 2=ปานกลาง, 3=รุนแรง)
                      → ตรงกับเป้าหมายหลักของงาน และเทียบกับงานรุ่นพี่ได้

  task="binary"      ยุบเหลือ 2 ระดับ (0=ไม่เสียหาย, 1=เสียหาย [รวม grade 1,2,3])
                      → แก้ปัญหาข้อมูลไม่สมดุล (จาก 40:1 เหลือ 11:1,
                         คลาสน้อยสุดจาก 270 ปล้อง เพิ่มเป็น 978 ปล้อง)

สำคัญ: การแบ่งข้อมูล (split) ใช้ชุดเดิมทั้ง 2 โจทย์ — ยุบ label ตอนโหลดข้อมูลตรงนี้
ไม่ใช่ตอนแบ่ง เพราะถ้าแบ่งใหม่คนละแบบ จะเทียบผล 2 โจทย์กันไม่แฟร์

load_split_csv() ตรวจสอบข้อมูลตั้งแต่ตอนโหลด (level_index อยู่ในช่วง 1-15 ไหม,
ไฟล์รูปมีอยู่จริงไหม) — ถ้าเจอปัญหาจะ raise error ทันที (fail fast) ไม่ปล่อยให้
ไปพังกลางการเทรนซึ่งเสียเวลามากกว่า ตั้ง strict=False ถ้าอยากแค่เตือนแล้วตัดทิ้งแทน

VertebraDataset.__getitem__ คืนค่า 4 อย่าง (image, level_idx, label, grade_4class)
ไม่ใช่ 3 อย่าง — grade_4class ติดมาด้วยเสมอเพื่อวิเคราะห์ผลตอนโหมด binary ได้โดยตรง
ไม่ต้องพึ่งลำดับแถวใน dataframe ที่เปราะบางถ้ามีการ shuffle
"""

# --- import library ที่ต้องใช้ ---
from pathlib import Path                   # เช็คว่าไฟล์รูปมีอยู่จริงไหม

import numpy as np                        # จัดการตัวเลข
import pandas as pd                        # จัดการตาราง
import torch                               # แปลงข้อมูลเป็น tensor ที่โมเดลใช้
from torch.utils.data import Dataset       # โครงมาตรฐานของ PyTorch สำหรับชุดข้อมูล

# import ฟังก์ชันเตรียมรูปจากไฟล์ transforms.py ที่อยู่โฟลเดอร์เดียวกัน
# จุด (.) นำหน้าแปลว่า "หาจากโฟลเดอร์เดียวกันกับไฟล์นี้"
from .transforms import prepare_image


# --- ตารางตั้งค่าของแต่ละโจทย์ ---
# เก็บไว้ที่เดียว เวลาจะเพิ่มโจทย์ใหม่ก็มาเพิ่มตรงนี้ ไม่ต้องแก้โค้ดข้างล่าง
TASKS = {
    "multiclass": {
        # แปลง grade ดิบจาก Excel เป็นตัวเลขที่โมเดลใช้
        # "4" คือค่าพิมพ์ผิดใน Excel (Genant มีแค่ 0-3) ถือเป็น 3 (รุนแรง)
        "label_map": {"0": 0, "1": 1, "2": 2, "3": 3, "4": 3},
        "num_classes": 4,
        "class_names": ["normal", "mild", "moderate", "severe"],
    },
    "binary": {
        # ยุบ: grade 0 อยู่คลาส 0, ส่วน grade 1/2/3 (และ 4 ที่เป็น typo) รวมเป็นคลาส 1
        "label_map": {"0": 0, "1": 1, "2": 1, "3": 1, "4": 1},
        "num_classes": 2,
        "class_names": ["undamaged", "damaged"],
    },
}

# ตารางแปลง grade ดิบเป็น 4 คลาส — เก็บไว้ต่างหากเพื่อใช้วิเคราะห์ผลตอนโหมด binary
# (จะได้ดูได้ว่าโมเดลจับ "เล็กน้อย" ได้จริงไหม หรือจับแค่ตัวที่ชัดๆ)
GRADE4_MAP = {"0": 0, "1": 1, "2": 2, "3": 3, "4": 3}


# ============================================================================
# ส่วน metadata (ข้อมูลผู้ป่วย: อายุ เพศ น้ำหนัก ส่วนสูง)
# ============================================================================
#
# ข้อควรรู้ก่อนใช้: metadata เป็นข้อมูล "ระดับคนไข้" แต่โจทย์เป็น "ระดับปล้อง"
# คนไข้ 1 คนมี 15 ปล้องที่ใช้ metadata ชุดเดียวกันหมด แปลว่า metadata บอกได้แค่
# "คนนี้มีแนวโน้มกระดูกหักแค่ไหน" แต่บอกไม่ได้ว่า "ปล้องไหนหัก" — ซึ่งเป็นสิ่งที่
# โจทย์ถามจริงๆ ใช้ได้แต่ต้องตีความผลอย่างระวัง

# คอลัมน์ใน Excel ที่จะเอามาใช้
METADATA_SOURCE_COLS = {
    "age": "Age at exam (years)",
    "sex": "Sex",
    "weight": "Weight (Kg)",
    "height": "Height (cm)",
}
METADATA_COLS = ["age", "sex", "weight", "height"]   # ชื่อที่ใช้ภายในโค้ด
NUM_METADATA = len(METADATA_COLS)

# ช่วงค่าที่เป็นไปได้ทางสรีรวิทยา — ค่านอกช่วงนี้ถือว่ากรอกผิด ต้องตัดทิ้ง
# (เจอจริงในข้อมูล: คนไข้ 1 รายกรอกน้ำหนัก 154.5 กก. ส่วนสูง 54.5 ซม. = สลับกัน
#  ถ้าปล่อยไว้จะทำให้ค่าเฉลี่ย/ส่วนเบี่ยงเบนมาตรฐานเพี้ยนทั้งชุด)
METADATA_VALID_RANGE = {
    "age": (10, 110),
    "weight": (20, 200),
    "height": (100, 210),
}


def load_metadata(xlsx_path: str, id_width: int = 4) -> pd.DataFrame:
    """
    อ่าน metadata ผู้ป่วยจาก DataTable.xlsx

    input:  xlsx_path = ที่อยู่ไฟล์ Excel
    output: ตาราง index = patient_id, คอลัมน์ = age, sex, weight, height
            (sex แปลงเป็นตัวเลข: หญิง=0, ชาย=1)
    """
    df = pd.read_excel(xlsx_path, sheet_name="Main")
    df["patient_id"] = df["No"].astype(int).astype(str).str.zfill(id_width)

    out = pd.DataFrame({"patient_id": df["patient_id"]})
    for name, src in METADATA_SOURCE_COLS.items():
        out[name] = df[src]

    # แปลงเพศเป็นตัวเลข (โมเดลรับได้แต่ตัวเลข)
    # ข้อสังเกต: ข้อมูลชุดนี้มีหญิง 804 : ชาย 29 — เกือบเป็นค่าคงที่
    # ใส่ได้แต่คาดหวังไม่ได้มากว่าจะช่วย
    out["sex"] = (out["sex"].astype(str).str.upper() == "M").astype(float)

    # --- ตัดค่าที่เป็นไปไม่ได้ทางสรีรวิทยาออก (ทำเป็น NaN แล้วเติมด้วยค่ากลางทีหลัง) ---
    for col, (lo, hi) in METADATA_VALID_RANGE.items():
        bad = ~out[col].between(lo, hi)
        if bad.any():
            bad_ids = out.loc[bad, "patient_id"].tolist()
            print(f"  เตือน: {col} มีค่านอกช่วง {lo}-{hi} จำนวน {bad.sum()} ราย "
                  f"(patient_id: {bad_ids[:5]}{'...' if len(bad_ids) > 5 else ''}) "
                  f"-> จะแทนด้วยค่ามัธยฐาน")
            out.loc[bad, col] = np.nan

    return out.set_index("patient_id")


def attach_metadata(df: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    """
    เชื่อม metadata เข้ากับตารางข้อมูลระดับปล้อง (join ด้วย patient_id)

    ปล้องทั้ง 15 ของคนเดียวกันจะได้ metadata ชุดเดียวกันหมด
    """
    df = df.copy()
    for col in METADATA_COLS:
        df[f"meta_{col}"] = df["patient_id"].map(metadata[col])
    return df


def compute_metadata_stats(train_df: pd.DataFrame) -> dict:
    """
    คำนวณค่าสถิติสำหรับปรับสเกล metadata — ใช้ "ชุด train เท่านั้น"

    สำคัญมาก: ห้ามคำนวณจากข้อมูลทั้งหมด (รวม val/test) เพราะจะเป็นการรั่วไหล
    ข้อมูล — ค่าเฉลี่ยของชุดทดสอบจะแอบส่งผลต่อการเทรน ทำให้ผลดูดีเกินจริง

    output: dict เก็บ median (ไว้เติมค่าที่ขาดหาย), mean, std ของแต่ละคอลัมน์
    """
    stats = {}
    for col in METADATA_COLS:
        vals = train_df[f"meta_{col}"].astype(float)
        median = float(vals.median())
        filled = vals.fillna(median)          # เติมค่าที่ขาดก่อนคำนวณ mean/std
        std = float(filled.std())
        stats[col] = {
            "median": median,
            "mean": float(filled.mean()),
            # กัน std เป็น 0 (เกิดได้ถ้าคอลัมน์นั้นมีค่าเดียวทั้งชุด) จะทำให้หารด้วยศูนย์
            "std": std if std > 1e-6 else 1.0,
        }
    return stats


def normalize_metadata_row(row, stats: dict) -> np.ndarray:
    """
    แปลง metadata ของ 1 แถว เป็นเวกเตอร์ตัวเลขที่ปรับสเกลแล้ว

    ใช้สูตร z-score: (ค่า - ค่าเฉลี่ย) / ส่วนเบี่ยงเบนมาตรฐาน
    เหตุผลเดียวกับการ normalize รูป — ให้ทุก feature อยู่ในสเกลใกล้เคียงกัน
    ไม่งั้น "ส่วนสูง 160" จะมีอิทธิพลมากกว่า "เพศ 0/1" มหาศาลโดยไม่มีเหตุผล
    """
    out = np.zeros(NUM_METADATA, dtype=np.float32)
    for i, col in enumerate(METADATA_COLS):
        v = row[f"meta_{col}"]
        s = stats[col]
        if pd.isna(v):
            v = s["median"]          # ค่าขาดหาย -> เติมด้วยมัธยฐานของชุด train
        out[i] = (float(v) - s["mean"]) / s["std"]
    return out


def load_split_csv(csv_path: str, task: str = "multiclass", strict: bool = True) -> pd.DataFrame:
    """
    อ่านไฟล์ split (เช่น xray_bbox_train.csv) แล้วเติมคอลัมน์ label ตามโจทย์ที่เลือก
    พร้อมตรวจสอบข้อมูลตั้งแต่ตอนโหลด (fail fast) แทนที่จะปล่อยให้ไปพังกลางการเทรน

    input:  csv_path = ที่อยู่ไฟล์ split ที่ run_split.py สร้างไว้
            task = "multiclass" หรือ "binary"
            strict = True (ค่าเริ่มต้น) -> เจอปัญหาแล้วหยุดทันที (raise error)
                     False -> แค่เตือนแล้วตัดแถวที่มีปัญหาทิ้ง ไปต่อได้
    output: ตาราง pandas พร้อมคอลัมน์:
            label        = คำตอบตามโจทย์ที่เลือก (0-3 หรือ 0-1)
            grade_4class = grade เดิม 4 ระดับ (เก็บไว้วิเคราะห์ผลตอนใช้โหมด binary)
    """
    # เช็คก่อนว่า task ที่ใส่มาถูกต้องไหม (ถ้าพิมพ์ผิดจะได้รู้ทันที ไม่ใช่ไปพังทีหลัง)
    if task not in TASKS:
        raise ValueError(f"task ต้องเป็น {list(TASKS)} เท่านั้น แต่ได้รับ '{task}'")

    label_map = TASKS[task]["label_map"]   # ดึงตารางแปลง label ของโจทย์นี้

    # อ่าน CSV โดยบังคับชนิดข้อมูล 2 คอลัมน์
    #   grade_raw: อ่านเป็น string เพราะมีทั้งตัวเลขและตัวอักษร ("x") ปนกัน
    #   patient_id: อ่านเป็น string กัน pandas ตัดเลข 0 นำหน้าทิ้ง (0002 -> 2)
    df = pd.read_csv(csv_path, dtype={"grade_raw": str, "patient_id": str})

    # ทำความสะอาดค่า: ช่องว่างเปล่าเป็น string ว่าง, ตัดเว้นวรรคหน้า-หลังออก
    df["grade_raw"] = df["grade_raw"].fillna("").str.strip()

    # กรองเหลือเฉพาะแถวที่แปลงเป็น label ได้ตามโจทย์นี้ (99/x ถูกกรองออกไปแล้วตอน split
    # เป็นปกติ แต่กันไว้อีกชั้นเผื่อไฟล์ split มีค่าที่แปลงไม่ได้หลงเหลืออยู่)
    keep = df["grade_raw"].isin(label_map.keys())
    df = df[keep].copy()

    # เติมคอลัมน์คำตอบตามโจทย์ที่เลือก
    df["label"] = df["grade_raw"].map(label_map).astype(int)

    # เติมคอลัมน์ grade เดิม 4 ระดับไว้ด้วยเสมอ (ไม่ว่าจะเลือกโจทย์ไหน)
    # ตอนโหมด binary จะได้เอามาแยกดูว่า "ที่ทายว่าเสียหายน่ะ จับ grade ไหนได้บ้าง"
    df["grade_4class"] = df["grade_raw"].map(GRADE4_MAP).astype(int)

    # --- ตรวจสอบที่ 1: level_index ต้องอยู่ในช่วง 1-15 เท่านั้น ---
    # ถ้าหลุดช่วงนี้ไป ตอนแปลงเป็น level_idx (ลบ 1) แล้วป้อนเข้า nn.Embedding(15, ...)
    # จะ error ทันที แต่จะพังตอนเทรน (กลางทาง) แทนที่จะพังตอนโหลด (ตั้งแต่ต้น)
    bad_level = ~df["level_index"].between(1, 15)
    if bad_level.any():
        bad_rows = df[bad_level]
        msg = (f"เจอ level_index นอกช่วง 1-15 จำนวน {bad_level.sum()} แถว "
              f"เช่น patient_id={bad_rows.iloc[0]['patient_id']} "
              f"level_index={bad_rows.iloc[0]['level_index']}")
        if strict:
            raise ValueError(msg)   # หยุดทันที ให้คนไปเช็คไฟล์ split/manifest ต้นทาง
        else:
            print(f"WARNING: {msg} — ตัดแถวเหล่านี้ทิ้ง")
            df = df[~bad_level].copy()

    # --- ตรวจสอบที่ 2: ไฟล์รูปต้องมีอยู่จริงทุกแถว ---
    # ถ้าไฟล์หาย DataLoader จะ error ตอนเทรน (อาจกลางดึกหลังรันไปหลาย epoch แล้ว)
    # เช็คตอนนี้ทีเดียว รู้ผลก่อนเสียเวลาเทรนไปเปล่าๆ
    file_exists = df["crop_path"].apply(lambda p: Path(p).exists())
    missing = df[~file_exists]
    if len(missing) > 0:
        msg = (f"เจอไฟล์รูปที่ไม่มีอยู่จริง {len(missing)} แถว "
              f"เช่น {missing.iloc[0]['crop_path']}")
        if strict:
            raise FileNotFoundError(msg)
        else:
            print(f"WARNING: {msg} — ตัดแถวเหล่านี้ทิ้ง")
            df = df[file_exists].copy()

    return df.reset_index(drop=True)   # เรียงเลขลำดับแถวใหม่ 0,1,2,...


class VertebraDataset(Dataset):
    """
    ตัวป้อนข้อมูลให้โมเดล — PyTorch จะเรียกใช้ซ้ำๆ ระหว่างเทรน สุ่มหยิบทีละแถว

    สืบทอด (inherit) จาก Dataset ของ PyTorch แปลว่าต้องมี 2 เมธอดนี้เสมอ:
      __len__     บอกว่ามีข้อมูลทั้งหมดกี่ตัวอย่าง
      __getitem__ บอกว่า "ถ้าขอตัวอย่างลำดับที่ idx จะได้อะไรกลับไป"
    """

    def __init__(self, df: pd.DataFrame, backbone: str = "efficientnet_b0",
                 img_size: int = None, metadata_stats: dict = None,
                 resize_mode: str = "pad"):
        """
        df = ตารางจาก load_split_csv()
        backbone = ชื่อ backbone ที่จะเทรนด้วย (ใช้เลือกวิธี normalize ให้ตรงกัน)
        img_size = ขนาดรูปที่จะป้อนโมเดล — ไม่ใส่ (None, ค่าเริ่มต้น) จะได้ขนาดมาตรฐาน
                   ของ backbone นั้นอัตโนมัติจาก transforms.py (224 ทั่วไป, 518 สำหรับ rad_dino)
        metadata_stats = ค่าสถิติสำหรับปรับสเกล metadata (จาก compute_metadata_stats)
                         ไม่ใส่ (None, ค่าเริ่มต้น) = ไม่ใช้ metadata, จะคืนเวกเตอร์ศูนย์แทน
        resize_mode = "pad" (ค่าเริ่มต้น, พฤติกรรมเดิม) เติมขอบดำ รักษาสัดส่วนกระดูก
                      "stretch" ยืดเต็มกรอบ ไม่รักษาสัดส่วน (ใช้ได้กับ backbone ทั่วไปเท่านั้น)

        ไม่มีตัวเลือกดัดแปลงรูป — ทุกชุด (train/val/test) เตรียมรูปเหมือนกันหมด
        คือย่อ+ปรับสเกลเท่านั้น ไม่มีการแต่งภาพใดๆ
        """
        self.df = df.reset_index(drop=True)   # เก็บตารางไว้ใช้ (self = ตัวแปรของ object นี้)
        self.backbone = backbone
        self.img_size = img_size   # อาจเป็น None -> ให้ transforms.py เลือก default ให้เอง
        self.metadata_stats = metadata_stats
        self.resize_mode = resize_mode

    def __len__(self):
        return len(self.df)   # จำนวนแถวในตาราง = จำนวนตัวอย่างทั้งหมด

    def __getitem__(self, idx):
        """
        คืนค่า 5 อย่างต่อ 1 ตัวอย่าง:
          image        = รูป crop ที่เตรียมแล้ว (สิ่งที่โมเดลดู)
          level_idx    = ปล้องที่เท่าไหร่ 0-14 (สิ่งที่บอกโมเดลเพิ่ม)
          metadata     = เวกเตอร์ 4 ค่า (อายุ เพศ น้ำหนัก ส่วนสูง) ที่ปรับสเกลแล้ว
                         ถ้าไม่ได้เปิดใช้ metadata จะเป็นเวกเตอร์ศูนย์ (โมเดลจะไม่สนใจอยู่แล้ว)
          label        = คำตอบที่ถูกต้องตามโจทย์ที่เลือก (ไม่ป้อนเข้าโมเดล ใช้เทียบตอนคำนวณความผิดพลาด)
          grade_4class = grade เดิม 4 ระดับเสมอ ไม่ว่าจะเลือกโจทย์ไหน (สำหรับตอนโหมด binary
                         จะได้แยกวิเคราะห์ได้ว่า "เสียหาย" ที่ทายถูก/ผิด เป็น grade ไหนบ้าง)
        """
        row = self.df.iloc[idx]   # .iloc[ตำแหน่ง] = ดึงแถวที่ตำแหน่งนี้ออกมา

        # เตรียมรูป (เรียกฟังก์ชันจาก transforms.py) — ส่ง backbone ไปด้วยให้เลือก normalize ถูกแบบ
        img = prepare_image(row["crop_path"], backbone=self.backbone, size=self.img_size,
                            resize_mode=self.resize_mode)

        # แปลง level จาก 1-15 (แบบที่คนอ่าน) เป็น 0-14 (แบบที่โมเดลใช้)
        # เพราะตารางค้นหาใน level embedding เริ่มนับจากแถวที่ 0 ไม่ใช่แถวที่ 1
        level_idx = int(row["level_index"]) - 1

        # metadata: ถ้าไม่ได้เปิดใช้ ให้คืนเวกเตอร์ศูนย์ (ขนาดคงที่เสมอ เพื่อให้ DataLoader
        # รวมเป็น batch ได้ไม่ว่าจะเปิดหรือปิดใช้งาน)
        if self.metadata_stats is not None:
            meta = normalize_metadata_row(row, self.metadata_stats)
        else:
            meta = np.zeros(NUM_METADATA, dtype=np.float32)

        # torch.from_numpy(...) = แปลง numpy array เป็น tensor ของ PyTorch
        return (torch.from_numpy(img), level_idx, torch.from_numpy(meta),
                int(row["label"]), int(row["grade_4class"]))


def compute_class_weights(df: pd.DataFrame, num_classes: int) -> torch.Tensor:
    """
    คำนวณ "น้ำหนักถ่วง" ให้แต่ละคลาส เพื่อชดเชยความไม่สมดุลของข้อมูล

    หลักการ: คลาสไหนมีตัวอย่างน้อย ให้น้ำหนักมาก (โมเดลจะได้ใส่ใจมากขึ้น)
    สูตร: น้ำหนัก = จำนวนทั้งหมด / (จำนวนคลาส × จำนวนตัวอย่างของคลาสนั้น)

    input:  df = ตารางชุด train (คำนวณจากชุด train เท่านั้น ห้ามใช้ val/test)
            num_classes = จำนวนคลาสของโจทย์ (4 หรือ 2)
    output: tensor ขนาดเท่าจำนวนคลาส เอาไปใส่ใน Focal Loss
    """
    # นับจำนวนแต่ละคลาส, .reindex(...) บังคับให้มีครบทุกคลาสแม้บางคลาสจะไม่มีเลย
    counts = df["label"].value_counts().reindex(range(num_classes)).fillna(0).values
    total = counts.sum()

    # np.clip(counts, 1, None) = ถ้าคลาสไหนนับได้ 0 ให้ใช้ 1 แทน (กันหารด้วยศูนย์)
    weights = total / (num_classes * np.clip(counts, 1, None))

    return torch.tensor(weights, dtype=torch.float32)