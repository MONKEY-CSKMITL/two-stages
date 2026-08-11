"""
sampling.py — สุ่มลดจำนวนปล้อง normal ในแต่ละ epoch (ไฟล์นี้เป็น "ห้องสมุด" ไม่ได้รันเอง)

แยกออกมาเป็นไฟล์ใหม่ ไม่แตะ dataset.py เดิมเลย ตามแนวทางเดียวกับ losses_ce.py

ปัญหาที่แก้: grade ปกติ (label=0) มีถึง 92% ของข้อมูล train ทำให้โมเดลเห็น
"ปกติ" ซ้ำๆ เยอะเกินไปเทียบกับปล้องที่หัก แม้จะมี Focal Loss ช่วยถ่วงน้ำหนักแล้ว
ก็ตาม — ไฟล์นี้แก้ที่ต้นตอ (ตัวข้อมูลที่โมเดลเห็นจริง) แทนที่จะแก้แค่ที่ loss

รองรับ 2 โหมด สลับได้ด้วยค่า mode ตัวเดียว:

  mode="dynamic" (ค่าเริ่มต้น — พฤติกรรมเดิมของไฟล์นี้ก่อนแก้ครั้งนี้)
    epoch 1: สุ่มปล้อง normal ชุด A มาใช้ (พร้อมปล้องที่หักทั้งหมด ไม่ตัดเลย)
    epoch 2: สุ่มปล้อง normal ชุด B ใหม่ (คนละชุดกับ A)
    epoch 3: สุ่มปล้อง normal ชุด C ใหม่ ...
    ทำแบบนี้เพื่อให้แต่ละ epoch ข้อมูล "สมดุล" ตามสัดส่วนที่ตั้งไว้ (เช่น 5:1)
    แต่พอรวมหลาย epoch โมเดลก็ยังมีโอกาสได้เห็นปล้อง normal เกือบครบทุกปล้อง
    ไม่ใช่ถูกจำกัดอยู่แค่กลุ่มเดียวตลอดการเทรน (ซึ่งเสี่ยงเห็นความหลากหลายของ
    กระดูกปกติไม่พอ)

  mode="fixed" (เพิ่มใหม่ — สุ่มครั้งเดียวตอนเริ่มเทรน แล้วใช้ชุดเดิมซ้ำทุก epoch)
    สุ่มปล้อง normal มาแค่ครั้งเดียวตอนสร้าง sampler แล้วใช้ "ชุดเดิมซ้ำทุก epoch"
    ตลอดการเทรน (ลำดับใน batch ยังสลับได้ทุก epoch เหมือนเดิม แค่ตัวปล้องที่ถูก
    เลือกมาไม่เปลี่ยนเลย) ข้อดี: เข้าใจง่ายตรงไปตรงมา เหมือนวิธี downsample แบบ
    ดั้งเดิมทั่วไปที่ตัดข้อมูลทิ้งครั้งเดียว — ข้อเสีย: เห็นความหลากหลายของ normal
    น้อยกว่า dynamic มาก เพราะ "ล็อก" อยู่กับกลุ่มตัวอย่างกลุ่มเดียวตลอดการเทรน

ใช้ได้กับทุกโจทย์ (multiclass, binary, 3class) เพราะ label=0 หมายถึง "ปกติ"
เหมือนกันในทุก label_map ที่มีอยู่ตอนนี้
"""

import numpy as np
from torch.utils.data import Sampler


class DownsampledNormalSampler(Sampler):
    """
    Sampler ที่บอก DataLoader ว่า "จะหยิบข้อมูลแถวไหนบ้างในรอบนี้"

    ปล้องที่หัก (label != 0) ใช้ทั้งหมดทุก epoch ไม่ตัดเลย (มีน้อยอยู่แล้ว)
    ปล้องปกติ (label == 0) สุ่มหยิบมาแค่บางส่วนตาม ratio ที่ตั้งไว้ — ส่วน "สุ่มใหม่
    ทุก epoch หรือสุ่มครั้งเดียว" คุมด้วย mode (ดู docstring หัวไฟล์)
    """

    def __init__(self, df, ratio: float, seed: int = 42, mode: str = "dynamic"):
        """
        df    = ตาราง train (ต้องมีคอลัมน์ "label" ที่ 0 = ปกติเสมอทุกโจทย์)
        ratio = สัดส่วน normal:fracture ที่ต้องการต่อ epoch เช่น 5.0 = 5:1
        seed  = ตัวเริ่มการสุ่ม (ล็อกไว้ให้ทำซ้ำได้ แต่ตอน mode="dynamic" ค่าที่สุ่ม
                ได้ยังต่างกันไปทุก epoch เพราะ RandomState เดินหน้าไปเรื่อยๆ ไม่ถูก
                reset กลับที่เดิมทุกครั้ง)
        mode  = "dynamic" (ค่าเริ่มต้น, พฤติกรรมเดิม) สุ่ม normal ใหม่ทุก epoch
                "fixed" สุ่มครั้งเดียวตอนสร้าง sampler แล้วใช้ชุดเดิมซ้ำทุก epoch
        """
        if mode not in ("dynamic", "fixed"):
            raise ValueError(f"mode ต้องเป็น 'dynamic' หรือ 'fixed' เท่านั้น ได้รับ '{mode}'")

        # .to_numpy() ดึงตำแหน่งแถว (ไม่ใช่ label) ของปล้องปกติ/ปล้องที่หัก แยกกัน
        self.normal_idx = df.index[df["label"] == 0].to_numpy()
        self.fracture_idx = df.index[df["label"] != 0].to_numpy()
        self.ratio = ratio
        self.mode = mode
        self.rng = np.random.RandomState(seed)   # เก็บไว้เป็น attribute -> เดินหน้าทุกครั้งที่เรียก __iter__

        # จำนวน normal ที่จะหยิบต่อ epoch คำนวณจาก ratio × จำนวนปล้องที่หักทั้งหมด
        # min(...) กันกรณี ratio สูงจนคำนวณได้เกินจำนวน normal ที่มีจริง
        target = int(len(self.fracture_idx) * ratio)
        self.n_normal_per_epoch = min(len(self.normal_idx), target)

        # โหมด fixed: สุ่มเลือกชุด normal "ครั้งเดียว" ตรงนี้เลย เก็บไว้ใช้ซ้ำทุก epoch
        # (ต่างจาก dynamic ที่จะสุ่มใหม่ทุกครั้งใน __iter__ แทน)
        self._fixed_normal = None
        if mode == "fixed":
            self._fixed_normal = self.rng.choice(self.normal_idx, size=self.n_normal_per_epoch,
                                                  replace=False)

        print(f"  DownsampledNormalSampler (mode={mode}): ratio={ratio}:1 -> "
              f"normal {self.n_normal_per_epoch}/{len(self.normal_idx)} ต่อ epoch "
              f"+ fracture {len(self.fracture_idx)} (ใช้ครบทุกอัน)")

    def __iter__(self):
        """
        PyTorch เรียกเมธอดนี้ใหม่ทุกครั้งที่เริ่มวน DataLoader รอบใหม่ (ทุก epoch)

        dynamic: สุ่มปล้อง normal ชุดใหม่ทุกครั้งที่ถูกเรียก เพราะ self.rng เดินหน้าต่อเนื่อง
        fixed:   ใช้ชุดที่สุ่มไว้ตอน __init__ ซ้ำทุกครั้ง ไม่สุ่มใหม่เลย
        """
        if self.mode == "fixed":
            chosen_normal = self._fixed_normal   # ชุดเดิมทุกครั้ง ไม่สุ่มใหม่
        else:
            # replace=False = สุ่มโดยไม่หยิบซ้ำภายในชุดเดียวกัน (ปล้องเดิมไม่โผล่ 2 ครั้งใน epoch เดียว)
            chosen_normal = self.rng.choice(self.normal_idx, size=self.n_normal_per_epoch, replace=False)

        # รวมปล้องปกติที่สุ่มมา + ปล้องที่หักทั้งหมด (ไม่ตัดเลย)
        epoch_idx = np.concatenate([chosen_normal, self.fracture_idx])

        # สลับลำดับ ไม่ให้ normal กับ fracture เรียงติดกันเป็นก้อนใหญ่ๆ — ทำทุก epoch
        # เสมอไม่ว่าโหมดไหน (fixed สลับแค่ "ลำดับ" ไม่ได้เปลี่ยน "ตัวที่ถูกเลือก")
        self.rng.shuffle(epoch_idx)

        return iter(epoch_idx.tolist())

    def __len__(self):
        """จำนวนตัวอย่างทั้งหมดต่อ epoch (PyTorch ใช้ค่านี้คำนวณจำนวน batch)"""
        return self.n_normal_per_epoch + len(self.fracture_idx)