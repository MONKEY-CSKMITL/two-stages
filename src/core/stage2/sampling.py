"""
sampling.py — สุ่มลดจำนวนปล้อง normal ในแต่ละ epoch (ไฟล์นี้เป็น "ห้องสมุด" ไม่ได้รันเอง)

แยกออกมาเป็นไฟล์ใหม่ ไม่แตะ dataset.py เดิมเลย ตามแนวทางเดียวกับ losses_ce.py

ปัญหาที่แก้: grade ปกติ (label=0) มีถึง 92% ของข้อมูล train ทำให้โมเดลเห็น
"ปกติ" ซ้ำๆ เยอะเกินไปเทียบกับปล้องที่หัก แม้จะมี Focal Loss ช่วยถ่วงน้ำหนักแล้ว
ก็ตาม — ไฟล์นี้แก้ที่ต้นตอ (ตัวข้อมูลที่โมเดลเห็นจริง) แทนที่จะแก้แค่ที่ loss

หลักการ "สุ่มใหม่ทุก epoch" (ไม่ใช่ตัดทิ้งครั้งเดียวตายตัว):
  epoch 1: สุ่มปล้อง normal ชุด A มาใช้ (พร้อมปล้องที่หักทั้งหมด ไม่ตัดเลย)
  epoch 2: สุ่มปล้อง normal ชุด B ใหม่ (คนละชุดกับ A)
  epoch 3: สุ่มปล้อง normal ชุด C ใหม่ ...
ทำแบบนี้เพื่อให้แต่ละ epoch ข้อมูล "สมดุล" ตามสัดส่วนที่ตั้งไว้ (เช่น 5:1)
แต่พอรวมหลาย epoch โมเดลก็ยังมีโอกาสได้เห็นปล้อง normal เกือบครบทุกปล้อง
ไม่ใช่ถูกจำกัดอยู่แค่กลุ่มเดียวตลอดการเทรน (ซึ่งเสี่ยงเห็นความหลากหลายของ
กระดูกปกติไม่พอ)

ใช้ได้กับทุกโจทย์ (multiclass, binary, 3class) เพราะ label=0 หมายถึง "ปกติ"
เหมือนกันในทุก label_map ที่มีอยู่ตอนนี้
"""

import numpy as np
from torch.utils.data import Sampler


class DownsampledNormalSampler(Sampler):
    """
    Sampler ที่บอก DataLoader ว่า "จะหยิบข้อมูลแถวไหนบ้างในรอบนี้"

    ปล้องที่หัก (label != 0) ใช้ทั้งหมดทุก epoch ไม่ตัดเลย (มีน้อยอยู่แล้ว)
    ปล้องปกติ (label == 0) สุ่มหยิบมาแค่บางส่วนตาม ratio ที่ตั้งไว้ และสุ่มใหม่
    ทุกครั้งที่ __iter__ ถูกเรียก (ซึ่ง PyTorch เรียกให้อัตโนมัติทุกครั้งที่เริ่ม
    epoch ใหม่ตอนวน for บน DataLoader — ไม่ต้องเขียนโค้ดเพิ่มเพื่อ "reset ทุก epoch"
    เอง เพราะเป็นพฤติกรรมมาตรฐานของ PyTorch DataLoader อยู่แล้ว)
    """

    def __init__(self, df, ratio: float, seed: int = 42):
        """
        df    = ตาราง train (ต้องมีคอลัมน์ "label" ที่ 0 = ปกติเสมอทุกโจทย์)
        ratio = สัดส่วน normal:fracture ที่ต้องการต่อ epoch เช่น 5.0 = 5:1
        seed  = ตัวเริ่มการสุ่ม (ล็อกไว้ให้ทำซ้ำได้ แต่ค่าที่สุ่มได้ยังต่างกันไปทุก epoch
                เพราะ RandomState เดินหน้าไปเรื่อยๆ ไม่ถูก reset กลับที่เดิมทุกครั้ง)
        """
        # .to_numpy() ดึงตำแหน่งแถว (ไม่ใช่ label) ของปล้องปกติ/ปล้องที่หัก แยกกัน
        self.normal_idx = df.index[df["label"] == 0].to_numpy()
        self.fracture_idx = df.index[df["label"] != 0].to_numpy()
        self.ratio = ratio
        self.rng = np.random.RandomState(seed)   # เก็บไว้เป็น attribute -> เดินหน้าทุกครั้งที่เรียก __iter__

        # จำนวน normal ที่จะหยิบต่อ epoch คำนวณจาก ratio × จำนวนปล้องที่หักทั้งหมด
        # min(...) กันกรณี ratio สูงจนคำนวณได้เกินจำนวน normal ที่มีจริง
        target = int(len(self.fracture_idx) * ratio)
        self.n_normal_per_epoch = min(len(self.normal_idx), target)

        print(f"  DownsampledNormalSampler: ratio={ratio}:1 -> "
              f"normal {self.n_normal_per_epoch}/{len(self.normal_idx)} ต่อ epoch "
              f"+ fracture {len(self.fracture_idx)} (ใช้ครบทุกอัน)")

    def __iter__(self):
        """
        PyTorch เรียกเมธอดนี้ใหม่ทุกครั้งที่เริ่มวน DataLoader รอบใหม่ (ทุก epoch)
        -> สุ่มปล้อง normal ชุดใหม่ทุกครั้งที่ถูกเรียก เพราะ self.rng เดินหน้าต่อเนื่อง
        """
        # replace=False = สุ่มโดยไม่หยิบซ้ำภายในชุดเดียวกัน (ปล้องเดิมไม่โผล่ 2 ครั้งใน epoch เดียว)
        chosen_normal = self.rng.choice(self.normal_idx, size=self.n_normal_per_epoch, replace=False)

        # รวมปล้องปกติที่สุ่มมา + ปล้องที่หักทั้งหมด (ไม่ตัดเลย)
        epoch_idx = np.concatenate([chosen_normal, self.fracture_idx])

        # สลับลำดับ ไม่ให้ normal กับ fracture เรียงติดกันเป็นก้อนใหญ่ๆ
        # (ถ้าไม่สลับ บาง batch อาจได้แต่ normal ล้วน บาง batch ได้แต่ fracture ล้วน)
        self.rng.shuffle(epoch_idx)

        return iter(epoch_idx.tolist())

    def __len__(self):
        """จำนวนตัวอย่างทั้งหมดต่อ epoch (PyTorch ใช้ค่านี้คำนวณจำนวน batch)"""
        return self.n_normal_per_epoch + len(self.fracture_idx)