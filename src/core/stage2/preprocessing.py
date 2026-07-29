"""
preprocessing.py — ที่เก็บฟังก์ชันแต่งภาพเพิ่มเติม (สำหรับทำ ablation ทีหลัง)

ทุกฟังก์ชันในไฟล์นี้มีรูปแบบเดียวกัน: รับภาพ PIL 1 ใบ คืนภาพ PIL 1 ใบกลับมา
เอาไปส่งให้ prepare_image(..., preprocess_fn=ฟังก์ชันในไฟล์นี้) ได้ทันที

ตอนนี้มีแค่ no_preprocess() ที่เป็น "ตัวว่าง" ไว้ทดสอบว่าการเสียบปลั๊กทำงานถูกก่อน
ฟังก์ชันอื่น (clahe, denoise) เป็นแค่โครงร่างรอเติมเนื้อหา ยังไม่ได้ทำอะไรจริง
"""

from PIL import Image


def no_preprocess(img: Image.Image) -> Image.Image:
    """
    ฟังก์ชันว่าง — ไม่ทำอะไรกับภาพเลย แค่คืนกลับไปเหมือนเดิม

    มีไว้เพื่อทดสอบว่า "เส้นทางที่มี preprocess_fn" ทำงานถูกต้อง โดยที่ผลลัพธ์
    ต้องเหมือนกับตอนที่ preprocess_fn=None เป๊ะ (เพราะฟังก์ชันนี้ไม่แก้อะไรเลย)
    ถ้าทดสอบแล้วผลไม่เหมือนกัน แปลว่าโค้ดจุดเสียบปลั๊กมีปัญหา
    """
    return img   # คืนภาพเดิม ไม่แตะต้องอะไรเลย


def clahe(img: Image.Image) -> Image.Image:
    """
    (ยังไม่ได้ทำ — โครงร่างรอเติม)
    เพิ่มความคมชัดเฉพาะจุด (Contrast Limited Adaptive Histogram Equalization)
    ตามที่รุ่นพี่รายงานว่าช่วยดันคลาส "เล็กน้อย" จาก 0% เป็น 47.5%
    """
    raise NotImplementedError("ยังไม่ได้เขียน — เติมโค้ด CLAHE ตรงนี้ตอนถึงเวลาทำ ablation")


def denoise(img: Image.Image) -> Image.Image:
    """
    (ยังไม่ได้ทำ — โครงร่างรอเติม)
    ลด noise ในภาพ X-ray
    """
    raise NotImplementedError("ยังไม่ได้เขียน — เติมโค้ด denoise ตรงนี้ตอนถึงเวลาทำ ablation")