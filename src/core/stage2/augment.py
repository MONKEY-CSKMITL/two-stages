"""
augment.py — การเพิ่มความหลากหลายให้ข้อมูลตอนเทรน (ไฟล์นี้เป็น "ห้องสมุด" ไม่ได้รันเอง)

--------------------------------------------------------------------------
ต่างจาก preprocessing.py อย่างไร — สำคัญมาก อย่าสับสน
--------------------------------------------------------------------------
                    preprocessing.py          augment.py (ไฟล์นี้)
  ใช้กับ split ไหน   train + val + test        train เท่านั้น
  สุ่มไหม            ไม่ (ผลเดิมทุกครั้ง)       สุ่มใหม่ทุกครั้งที่หยิบภาพ
  เป็นอะไร           "วิธีอ่านภาพ"              "การเพิ่มความหลากหลาย"

ถ้าเอา augmentation ไปใส่ใน preprocess_fn ชุด val/test จะถูกดัดแปลงแบบสุ่มไปด้วย
ทำให้วัดผลไม่ได้เลย — จึงต้องแยกเป็นคนละปลั๊กเด็ดขาด train.py จะส่ง augment_fn
ให้เฉพาะ dataset ของชุด train เท่านั้น (val/test ไม่ได้รับ จึงไม่มีทางถูก augment)

--------------------------------------------------------------------------
ทำไมต้องมี — และทำไมต้องเลือก transform อย่างระมัดระวังเป็นพิเศษกับงานนี้
--------------------------------------------------------------------------
ปัญหาที่วัดได้จาก log ของ baseline: train_loss ลงไปถึง 0.0001 ตั้งแต่ epoch ~20
(= จำภาพทั้ง 8,409 ใบได้หมด) ขณะที่ val_AUC ดีที่สุดตั้งแต่ epoch 2 แล้วลดลง
4.2 pp ตลอดทาง — คือ overfit เต็มรูปแบบ ซึ่งคาดได้เพราะ**ยังไม่มี augmentation
สักตัวเดียว**

แต่โจทย์นี้เลือก transform ตามสูตรมาตรฐานไม่ได้ เพราะเกณฑ์ Genant นิยาม grade
ด้วย **รูปทรง** โดยตรง (สัดส่วนความสูงหน้า/กลาง/หลัง) ต่างจากงาน classification
ทั่วไปที่ label ไม่ขึ้นกับรูปทรง วัดจากข้อมูลจริงได้ว่าฟีเจอร์รูปทรงตัวเดียว
(ar = สูง/กว้าง) แยกหักจากปกติได้ AUC 0.712 และ fill ได้ 0.741

  ❌ ห้ามใช้เด็ดขาด
     flip แนวนอน  = สลับด้านหน้า-หลังของกระดูก การยุบแบบ anterior wedge (พบบ่อย)
                    จะกลายเป็น posterior wedge (พบน้อยมาก) แต่ label ไม่เปลี่ยน
                    = สอนโมเดลว่าทิศทางไม่สำคัญ ทั้งที่มันคือหัวใจของการวินิจฉัย
     flip แนวตั้ง  = สลับ endplate บน-ล่าง ด้วยเหตุผลเดียวกัน
     scale / zoom  = ทำลาย ar และขนาดสัมบูรณ์ ซึ่งเป็นสัญญาณที่แรงที่สุดที่เรามี
     elastic /     = บิดรูปทรง = บิดคำตอบ ภาพที่ได้จะมี label ที่ผิดไปจากเดิม
     grid distort

  ✅ ใช้ได้ — รบกวน "ความเข้ม" ไม่แตะ "รูปทรง"
     brightness / contrast / gamma / noise

  ⚠️ ใช้ได้แบบจำกัด
     หมุนเล็กน้อย ±7° + เลื่อน ±5% — กระดูกสันหลังเอียงตามความโค้งอยู่แล้ว มุมเล็กๆ
     จึงสมจริง และการเลื่อนช่วยแก้ปัญหาที่ Grad-CAM พบว่าโมเดลไปมองมุมดำ (เพราะ
     กระดูกไม่ได้อยู่กลางกรอบเป๊ะทุกใบอีกต่อไป) แต่ใช้ได้เฉพาะกับ resize_mode="pad"
     ที่มีพื้นที่ดำรองรับ — ถ้าใช้ "stretch" การหมุนจะตัดมุมกระดูกหายทันที

--------------------------------------------------------------------------
ความแรงของแต่ละตัวมาจากไหน (ไม่ได้ตั้งลอยๆ)
--------------------------------------------------------------------------
หลักการ: ภาพที่ augment แล้วต้องยัง "เป็นไปได้จริง" สำหรับเครื่อง DXA เครื่องนี้
ถ้าแรงเกินความผันผวนตามธรรมชาติของชุดข้อมูล เท่ากับสอนให้โมเดลทนต่อสิ่งที่
ไม่มีวันเกิดขึ้นจริง ซึ่งเปลืองกำลังโมเดลไปเปล่าๆ

วัดจาก crop จริง 300 ใบได้ว่า:
  - ความสว่างเฉลี่ยของภาพ ต่างกันระหว่างคนไข้ sd = 42 ระดับ บนค่าเฉลี่ย 131 (~32%)
    -> ตั้ง brightness ที่ ±20% ซึ่งอยู่ในช่วงที่พบจริง ไม่เกินธรรมชาติ
  - ความเปรียบต่างภายในภาพ sd = 15.4 ระดับ
    -> ตั้ง noise ที่ sigma 2-8 ซึ่งต่ำกว่าโครงสร้างจริงชัดเจน ไม่กลบรายละเอียด

--------------------------------------------------------------------------
กติกาพื้นหลัง — ใช้ร่วมกับ preprocessing.py
--------------------------------------------------------------------------
ยึดกติกา 3 ข้อเดียวกับ preprocessing.py (คิดสถิติจากพิกเซลกระดูกเท่านั้น /
ทาพื้นหลังกลับเป็น 0 / พิกเซลกระดูกห้ามเป็น 0) เพราะทุก transform ทำลายมันได้หมด:
เพิ่มความสว่างแล้วพื้นหลังจากดำกลายเป็นเทา, โรย noise แล้วพื้นที่ที่ควรว่างมีลาย

จุดที่ต่างจาก preprocessing และพลาดง่ายที่สุด: **transform เชิงเรขาคณิตทำให้
ตัวกระดูกขยับ** mask เดิมจึงใช้ทาพื้นหลังไม่ได้อีก ต้องหมุน/เลื่อน mask ตามไป
ด้วยแล้วใช้ mask อันใหม่ (ดู random_shift_rotate)

--------------------------------------------------------------------------
เรื่องการสุ่มกับ DataLoader worker
--------------------------------------------------------------------------
ใช้ np.random ซึ่ง PyTorch เวอร์ชันใหม่ตั้ง seed ให้แต่ละ worker แยกกันเองอยู่แล้ว
(เป็นค่าที่ derive จาก seed หลัก + เลข worker + รอบของ epoch) จึงได้ทั้ง 2 อย่าง
พร้อมกัน: worker คนละตัวสุ่มไม่ซ้ำกันในรอบเดียวกัน และรันซ้ำด้วย seed เดิมได้ผลเดิม
"""

import numpy as np
from PIL import Image

# ใช้กติกาพื้นหลังชุดเดียวกับ preprocessing.py — import มาใช้ต่อ ไม่เขียนซ้ำ
# เพื่อกันกรณีแก้ที่หนึ่งแล้วลืมแก้อีกที่จนสองไฟล์ตีความพื้นหลังไม่ตรงกัน
from .preprocessing import BACKGROUND_VALUE, _split_gray, _merge_gray


def _finish(out: np.ndarray, bone: np.ndarray, mode: str) -> Image.Image:
    """
    ปิดท้ายทุก transform ด้วยกติกาข้อ 2 และ 3 (ทาพื้นหลัง / กันกระดูกเป็น 0)

    bone = ตารางบูลีนของพิกเซลกระดูก **หลัง** transform แล้ว (สำคัญกับ transform
           เชิงเรขาคณิตที่ตัวกระดูกขยับ ต้องส่ง mask อันใหม่มา ไม่ใช่อันเดิม)
    """
    out = np.clip(out, 1, 255)            # กติกา 3: พิกเซลกระดูกอยู่ในช่วง 1-255
    out[~bone] = BACKGROUND_VALUE          # กติกา 2: พื้นหลังกลับไปเป็น 0
    return _merge_gray(out, mode)


# ============================================================================
# กลุ่มปลอดภัย — รบกวนความเข้ม ไม่แตะรูปทรง
# ============================================================================

def random_brightness_contrast(img: Image.Image, brightness: float = 0.2,
                               contrast: float = 0.2, p: float = 0.5) -> Image.Image:
    """
    สุ่มปรับความสว่างและความเปรียบต่าง — จำลองความหนาตัวคนไข้/การตั้งค่าเครื่องที่ต่างกัน

    brightness/contrast = ขอบเขตการสุ่ม (0.2 = สุ่มในช่วง -20% ถึง +20%)
    p = โอกาสที่จะทำ (0.5 = ทำครึ่งหนึ่งของครั้งที่หยิบภาพ)

    หมุนรอบ "ค่าเฉลี่ยของพิกเซลกระดูก" ไม่ใช่รอบ 128 หรือ 0 เพราะภาพเรามีพื้นหลัง
    ดำก้อนใหญ่ ถ้าหมุนรอบค่าที่ไม่ใช่ค่ากลางของกระดูกจริง ผลจะเบ้ไปทางเดียวเสมอ
    """
    if np.random.rand() >= p:
        return img

    gray, mode = _split_gray(img)
    bone = gray > BACKGROUND_VALUE
    if not bone.any():
        return img

    b = np.random.uniform(-brightness, brightness)
    c = np.random.uniform(-contrast, contrast)

    center = float(gray[bone].mean())
    out = gray.astype(np.float32)
    out = center + (out - center) * (1.0 + c)   # ปรับความเปรียบต่างรอบค่ากลางของกระดูก

    # --- จำกัดการเลื่อนตาม "พื้นที่ว่างที่เหลือจริง" ของภาพใบนั้น ---
    # ปัญหาที่แก้: ภาพที่สว่างอยู่แล้ว (เช่นปล้องที่ความหนาแน่นกระดูกสูง) ถ้าโดน
    # เลื่อนขึ้นอีก 30-35% จะไปชนเพดาน 255 แล้วโดน clip ทิ้ง = รายละเอียดในเนื้อ
    # กระดูกหายเกลี้ยงกลายเป็นแผ่นขาว ซึ่งไม่ใช่ "ภาพที่เป็นไปได้จริง" อีกต่อไป
    # และเป็นการทำลายข้อมูล ไม่ใช่การเพิ่มความหลากหลาย (เห็นชัดตอนตรวจด้วยตา)
    #
    # แก้โดยดูว่าภาพใบนี้ยังเลื่อนขึ้น/ลงได้อีกเท่าไหร่ก่อนจะชนขอบ แล้วบีบค่าที่
    # สุ่มได้ให้อยู่ในช่วงนั้น — ภาพที่มีพื้นที่ว่างเยอะก็ยังได้ความหลากหลายเต็มที่
    # ส่วนภาพที่สว่าง/มืดจัดอยู่แล้วจะถูกรบกวนน้อยลงโดยอัตโนมัติ
    vals = out[bone]
    room_up = 250.0 - float(np.percentile(vals, 99))    # เลื่อนขึ้นได้อีกเท่าไหร่
    room_down = float(np.percentile(vals, 1)) - 5.0     # เลื่อนลงได้อีกเท่าไหร่
    shift = float(np.clip(b * center, -max(room_down, 0.0), max(room_up, 0.0)))
    out = out + shift

    return _finish(out, bone, mode)


def random_gamma(img: Image.Image, lo: float = 0.85, hi: float = 1.15,
                 p: float = 0.3) -> Image.Image:
    """
    สุ่มปรับ gamma — จำลองการตอบสนองต่อความเข้มของ detector ที่ต่างรุ่นกัน

    ต่างจาก brightness ตรงที่ gamma เป็นการแปลงแบบ**ไม่เชิงเส้น** — ดันโทนมืดกับ
    โทนสว่างคนละทิศ จึงเพิ่มความหลากหลายคนละแบบกับ brightness/contrast ไม่ซ้ำซ้อน
    """
    if np.random.rand() >= p:
        return img

    gray, mode = _split_gray(img)
    bone = gray > BACKGROUND_VALUE
    if not bone.any():
        return img

    g = np.random.uniform(lo, hi)
    out = 255.0 * np.power(gray.astype(np.float32) / 255.0, g)

    return _finish(out, bone, mode)


def random_noise(img: Image.Image, sigma_lo: float = 2.0, sigma_hi: float = 8.0,
                 p: float = 0.3) -> Image.Image:
    """
    สุ่มโรย noise แบบเกาส์เซียน — จำลอง dose ที่ต่างกันของแต่ละครั้งที่สแกน

    sigma อยู่ที่ 2-8 ระดับความสว่าง ซึ่งต่ำกว่าความเปรียบต่างภายในภาพจริง (sd 15.4)
    อย่างชัดเจน จึงไม่กลบโครงสร้างของกระดูกที่โมเดลต้องใช้ตัดสิน

    โรยเฉพาะบนพิกเซลกระดูก — พื้นหลังต้องว่างเปล่าเสมอ ถ้าโรย noise ลงพื้นหลังด้วย
    จะกลายเป็นการสร้างสัญญาณปลอมในบริเวณที่ไม่ควรมีข้อมูลอะไรเลย
    """
    if np.random.rand() >= p:
        return img

    gray, mode = _split_gray(img)
    bone = gray > BACKGROUND_VALUE
    if not bone.any():
        return img

    sigma = np.random.uniform(sigma_lo, sigma_hi)
    out = gray.astype(np.float32) + np.random.normal(0.0, sigma, size=gray.shape)

    return _finish(out, bone, mode)


# ============================================================================
# กลุ่มจำกัด — ขยับตำแหน่ง/มุม แต่ไม่เปลี่ยนสัดส่วน
# ============================================================================

def random_shift_rotate(img: Image.Image, max_deg: float = 7.0,
                        max_shift: float = 0.05, p: float = 0.5) -> Image.Image:
    """
    สุ่มหมุนเล็กน้อยและเลื่อนตำแหน่ง — **ไม่มีการย่อ/ขยาย** โดยตั้งใจ

    ทำไมหมุนได้ แต่ย่อขยายไม่ได้: การหมุนไม่เปลี่ยนสัดส่วนความสูงต่อความกว้างของ
    ตัวกระดูกเอง (แค่วางเอียงในกรอบ) และกระดูกสันหลังจริงก็เอียงต่างกันตามความโค้ง
    ของแนวกระดูกอยู่แล้ว แต่การย่อขยายเปลี่ยน ar และขนาดสัมบูรณ์โดยตรง ซึ่งคือ
    สัญญาณที่ Genant ใช้ตัดสิน

    ต้องเรียกฟังก์ชันนี้ **หลัง** ขั้น resize/pad แล้วเท่านั้น เพราะ crop ตัดชิดตัว
    กระดูกพอดี ถ้าหมุนตอนยังไม่ pad มุมกระดูกจะถูกตัดหายไป

    จุดสำคัญที่ต่างจาก transform กลุ่มความเข้ม: ตัวกระดูกขยับ mask เดิมจึงใช้ไม่ได้
    ต้องหมุน mask ไปด้วยด้วยการแปลงชุดเดียวกันเป๊ะ แล้วใช้ mask อันใหม่ทาพื้นหลัง
    """
    if np.random.rand() >= p:
        return img

    gray, mode = _split_gray(img)
    bone = gray > BACKGROUND_VALUE
    if not bone.any():
        return img

    angle = np.random.uniform(-max_deg, max_deg)
    h, w = gray.shape
    dx = int(round(np.random.uniform(-max_shift, max_shift) * w))
    dy = int(round(np.random.uniform(-max_shift, max_shift) * h))

    # แปลงภาพ — fillcolor=0 ให้พื้นที่ที่หมุนเข้ามาใหม่เป็นพื้นหลังตามกติกา
    src = Image.fromarray(gray.astype(np.uint8), mode="L")
    moved = src.rotate(angle, resample=Image.BILINEAR, translate=(dx, dy), fillcolor=0)

    # แปลง mask ด้วยการหมุน/เลื่อนชุดเดียวกัน แล้วตัดที่ 127 ให้กลับเป็นบูลีน
    # (ใช้ BILINEAR เหมือนกันเพื่อให้ขอบของภาพกับของ mask ตรงกัน ไม่เหลื่อมกัน 1 พิกเซล)
    mask_src = Image.fromarray((bone * 255).astype(np.uint8), mode="L")
    mask_moved = mask_src.rotate(angle, resample=Image.BILINEAR, translate=(dx, dy), fillcolor=0)
    new_bone = np.array(mask_moved) > 127

    return _finish(np.array(moved).astype(np.float32), new_bone, mode)


# ============================================================================
# กลุ่มที่เอกสารด้านบนระบุว่า "ห้ามใช้" — เขียนไว้เพื่อ**ทดสอบข้อห้ามนั้นด้วยข้อมูล**
# ============================================================================
#
# ทำไมถึงมีอยู่ทั้งที่หัวไฟล์เขียนว่าห้าม: ข้อห้ามนั้นมาจากการให้เหตุผลทางคลินิก
# (Genant นิยาม grade ด้วยรูปทรง) ไม่ได้มาจากการทดลอง — ซึ่งเป็นเหตุผลที่ดี แต่
# ยังไม่ใช่หลักฐาน การมีฟังก์ชันพวกนี้ทำให้เปลี่ยน "ข้อสันนิษฐาน" เป็น "ผลที่วัดได้"
# ในเล่มได้ แทนที่จะเขียนว่า "ไม่ได้ลองเพราะคิดว่าไม่ควร"
#
# ⚠️ ทั้ง 3 ตัวนี้เปลี่ยน "รูปทรง" ซึ่งเป็นสิ่งที่ label อ้างอิงโดยตรง ต่างจากทุก
# ฟังก์ชันข้างบนที่ตั้งใจไม่แตะรูปทรง ผลที่ตามมาคือภาพที่ได้อาจมี label ที่ผิดไปแล้ว:
#   flip แนวนอน  anterior wedge (พบบ่อย) -> posterior wedge (พบน้อยมาก) label เท่าเดิม
#   flip แนวตั้ง  สลับ endplate บน-ล่าง ด้วยเหตุผลเดียวกัน
#   scale        เปลี่ยน ar และขนาดสัมบูรณ์ ซึ่ง ar เดี่ยวๆ แยกหักจากปกติได้ AUC 0.712
#   elastic      บิดสัดส่วนความสูงหน้า/กลาง/หลัง = บิดตัวเกณฑ์ที่ใช้ตัดสินเอง
# ถ้าใช้แล้วผลแย่ลง นั่นคือหลักฐานยืนยันข้อห้าม ไม่ใช่ความผิดพลาดของการทดลอง

def random_flip(img: Image.Image, p_h: float = 0.5, p_v: float = 0.5) -> Image.Image:
    """
    สุ่มพลิกภาพซ้าย-ขวา และ/หรือ บน-ล่าง

    p_h / p_v = โอกาสพลิกแต่ละแกน (แยกกัน ภาพหนึ่งใบพลิกได้ทั้ง 2 แกน)

    ไม่ต้องจัดการ mask แยก เพราะการพลิกไม่สร้างพิกเซลใหม่ ไม่มี interpolation
    พื้นหลังที่เป็น 0 ก็ยังเป็น 0 หลังพลิก — ต่างจาก rotate/scale/elastic ที่ต้อง
    แปลง mask ตามไปด้วย
    """
    do_h = np.random.rand() < p_h
    do_v = np.random.rand() < p_v
    if not (do_h or do_v):
        return img

    out = img
    if do_h:
        out = out.transpose(Image.FLIP_LEFT_RIGHT)
    if do_v:
        out = out.transpose(Image.FLIP_TOP_BOTTOM)
    return out


def random_scale(img: Image.Image, lo: float = 0.85, hi: float = 1.15,
                 p: float = 0.5) -> Image.Image:
    """
    สุ่มย่อ/ขยายรอบจุดกึ่งกลาง โดยกรอบภาพคงขนาดเดิม

    lo/hi = ช่วงตัวคูณขนาด (0.85 = เล็กลง 15%, 1.15 = ใหญ่ขึ้น 15%)

    ย่อขยาย **เท่ากันทั้ง 2 แกน** จึงไม่เปลี่ยน ar ของตัวกระดูกเอง แต่เปลี่ยน
    "ขนาดสัมบูรณ์เทียบกับกรอบ" ซึ่งเป็นสัญญาณที่โมเดลใช้ได้จริง (ปล้องที่ยุบจะเตี้ย
    กว่าปกติเมื่อเทียบกับกรอบที่ pad มาเท่ากัน) — การย่อขยายจึงลบสัญญาณนั้นทิ้ง

    ใช้ Image.transform แบบ AFFINE แทน resize+crop เพราะทำ mask ด้วยเมทริกซ์ชุด
    เดียวกันเป๊ะได้ ไม่ต้องกังวลว่าการปัดเศษของ 2 เส้นทางจะเหลื่อมกัน 1 พิกเซล
    """
    if np.random.rand() >= p:
        return img

    gray, mode = _split_gray(img)
    bone = gray > BACKGROUND_VALUE
    if not bone.any():
        return img

    s = float(np.random.uniform(lo, hi))
    h, w = gray.shape
    cx, cy = w / 2.0, h / 2.0
    # PIL AFFINE คิดย้อนทาง: พิกเซลผลลัพธ์ (x,y) ไปดึงค่าจากตำแหน่ง (a*x+b*y+c, d*x+e*y+f)
    # ของภาพต้นทาง จึงต้องใส่ 1/s ไม่ใช่ s
    inv = 1.0 / s
    matrix = (inv, 0.0, cx - cx * inv, 0.0, inv, cy - cy * inv)

    src = Image.fromarray(gray.astype(np.uint8), mode="L")
    scaled = src.transform((w, h), Image.AFFINE, matrix,
                           resample=Image.BILINEAR, fillcolor=0)

    mask_src = Image.fromarray((bone * 255).astype(np.uint8), mode="L")
    mask_scaled = mask_src.transform((w, h), Image.AFFINE, matrix,
                                     resample=Image.BILINEAR, fillcolor=0)
    new_bone = np.array(mask_scaled) > 127

    return _finish(np.array(scaled).astype(np.float32), new_bone, mode)


def random_elastic(img: Image.Image, alpha: float = 8.0, sigma: float = 12.0,
                   p: float = 0.5) -> Image.Image:
    """
    สุ่มบิดภาพแบบยืดหยุ่น (elastic deformation)

    alpha = ระยะการเลื่อนสูงสุดของแต่ละพิกเซล (หน่วยพิกเซล) — ยิ่งมากยิ่งบิดแรง
    sigma = ความนุ่มของสนามการเลื่อน — ยิ่งมากยิ่งบิดเป็นคลื่นใหญ่ๆ ไม่ใช่ฟันปลา

    วิธีทำ: สุ่มเวกเตอร์การเลื่อนของทุกพิกเซล -> เกลี่ยด้วย gaussian ให้เพื่อนบ้าน
    เลื่อนไปทางเดียวกัน (ไม่งั้นภาพจะเละเป็นเม็ดๆ ไม่ใช่การบิด) -> ดึงค่าตามสนามนั้น

    ค่าเริ่มต้น alpha=8 sigma=12 บนภาพ 224 px — เลือกจากการกวาดค่าแล้วดูด้วยตา
    (ดู outputs/catalog_shape/elastic_sweep.png) เป็นค่าที่แรงที่สุดที่ "เส้น
    endplate ยังอ่านออกว่าเป็น endplate" ซึ่งเป็นเงื่อนไขที่ขาดไม่ได้ เพราะถ้าภาพ
    ไม่เหลือโครงสร้างให้วัด ผลที่ได้จะแปลว่า "ภาพพัง" ไม่ใช่ "การบิดรูปทรงมีผลเสีย"
    ซึ่งเป็นคนละข้อสรุปกัน

    ที่ไม่ใช้แรงกว่านี้: ที่ alpha=12 ขึ้นไป เงาของปล้องบิดจนวัดสัดส่วนความสูงไม่ได้
    ที่ alpha=20 ภาพกลายเป็นก้อนที่ไม่เหลือความเป็นกระดูก
    ที่ไม่ใช้เบากว่านี้: ที่ alpha=4 การบิดแทบมองไม่เห็น จะได้ผลเป็นศูนย์โดยอัตโนมัติ
    แล้วสรุปผิดว่า "elastic ไม่มีผล" ทั้งที่จริงคือ "แทบไม่ได้ทำ elastic"

    mask ต้องผ่านสนามการเลื่อนชุดเดียวกัน ไม่งั้นขอบกระดูกกับ mask จะไม่ตรงกัน
    """
    if np.random.rand() >= p:
        return img

    from scipy.ndimage import gaussian_filter, map_coordinates

    gray, mode = _split_gray(img)
    bone = gray > BACKGROUND_VALUE
    if not bone.any():
        return img

    h, w = gray.shape

    # ต้อง normalize สนามหลังเกลี่ยก่อนคูณ alpha — ห้ามคูณ alpha ตรงๆ
    # เหตุผล: gaussian_filter ลดแอมพลิจูดของสัญญาณสุ่มลงตาม sigma อย่างรุนแรง
    # (ที่ sigma=10 เหลือราว 1.6% ของเดิม) ถ้าคูณ alpha=20 ตรงๆ จะได้ระยะเลื่อนจริง
    # เพียง ~0.3 px = ไม่ได้บิดอะไรเลย แล้วจะสรุปผิดว่า "elastic ไม่มีผล"
    # ทั้งที่จริงคือ "ไม่ได้ทำ elastic" — normalize ทำให้ alpha มีความหมายตามที่เขียนไว้
    # จริงๆ คือระยะเลื่อนสูงสุดเป็นพิกเซล
    def _field():
        f = gaussian_filter(np.random.uniform(-1, 1, (h, w)), sigma, mode="constant")
        peak = float(np.abs(f).max())
        return f / peak * alpha if peak > 1e-8 else f

    dx, dy = _field(), _field()

    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    coords = np.array([(yy + dy).ravel(), (xx + dx).ravel()])

    out = map_coordinates(gray.astype(np.float32), coords, order=1,
                          mode="constant", cval=0.0).reshape(h, w)
    mask_out = map_coordinates(bone.astype(np.float32), coords, order=1,
                               mode="constant", cval=0.0).reshape(h, w)

    return _finish(out, mask_out > 0.5, mode)


# ============================================================================
# ชุดสำเร็จรูป — เอาไว้อ้างชื่อจาก config
# ============================================================================

def no_augment(img: Image.Image) -> Image.Image:
    """
    ตัวว่าง — ไม่ทำอะไรเลย ใช้ทดสอบว่าปลั๊ก augment ต่อถูกต้อง
    ผลต้องเหมือน augment_fn=None เป๊ะ (หลักการเดียวกับ no_preprocess)
    """
    return img


def augment_intensity(img: Image.Image) -> Image.Image:
    """
    เฉพาะกลุ่มปลอดภัย — ไม่แตะรูปทรงเลย ใช้ได้กับทั้ง resize_mode pad และ stretch

    เป็นชุดที่ควรลองก่อนเพื่อน เพราะถ้าได้ผลดีขึ้นจะสรุปได้สะอาดว่า "มาจากการ
    เพิ่มความหลากหลายของความเข้ม" ไม่มีตัวแปรเรื่องรูปทรงมาปน
    """
    img = random_brightness_contrast(img, 0.2, 0.2, p=0.5)
    img = random_gamma(img, 0.85, 1.15, p=0.3)
    img = random_noise(img, 2.0, 8.0, p=0.3)
    return img


def augment_geometric(img: Image.Image) -> Image.Image:
    """
    เฉพาะการหมุน/เลื่อน — ใช้ได้กับ resize_mode="pad" เท่านั้น

    แยกไว้เป็นชุดเดี่ยวเพื่อวัดผลของมันล้วนๆ โดยเฉพาะคำถามว่าช่วยแก้ปัญหา
    "โมเดลมองมุมดำ" ที่ Grad-CAM พบในการทดลองที่ 3 ได้จริงไหม
    """
    return random_shift_rotate(img, max_deg=7.0, max_shift=0.05, p=0.5)


def augment_strong(img: Image.Image) -> Image.Image:
    """
    เวอร์ชันแรงขึ้น — สำหรับกรณีที่ augment_standard เอาไม่อยู่

    เหตุผลที่ต้องมี: วัดแล้วพบว่า augment_standard ทำให้ความสว่างของภาพเดียวกัน
    แกว่ง sd ~12 ระดับ ขณะที่ความต่างตามธรรมชาติระหว่างคนไข้จริงอยู่ที่ sd ~38
    ระดับ คือ augment แรงแค่ 1 ใน 3 ของความผันผวนที่มีอยู่จริงในชุดข้อมูล

    เมื่อ overfit หนักถึงขั้น train_loss = 0.0001 การ augment ที่อ่อนกว่าธรรมชาติ
    อาจไม่ขยับอะไรเลย แล้วเราจะสรุปผิดว่า "augmentation ไม่ช่วย" ทั้งที่จริงคือ
    "augmentation ของเราเบาไป" ชุดนี้ตั้งให้เทียบเท่าความผันผวนธรรมชาติ

    ยังอยู่ในกรอบ "ภาพต้องเป็นไปได้จริง" เพราะอิงจากช่วงที่วัดได้จากข้อมูลเอง
    ไม่ได้เดาเอา — และยังไม่แตะกลุ่มที่ห้าม (flip / scale / distort) เหมือนเดิม
    """
    img = random_brightness_contrast(img, 0.35, 0.30, p=0.7)
    img = random_gamma(img, 0.75, 1.30, p=0.5)
    img = random_noise(img, 3.0, 12.0, p=0.5)
    img = random_shift_rotate(img, max_deg=10.0, max_shift=0.07, p=0.7)
    return img


def augment_standard(img: Image.Image) -> Image.Image:
    """
    ทั้ง 2 กลุ่มรวมกัน — ชุดที่คาดว่าจะแรงพอสำหรับปัญหา overfit ที่วัดได้

    เรียงความเข้มก่อนเรขาคณิต เพราะการหมุนใช้ interpolation ซึ่งเกลี่ยค่าพิกเซล
    ข้างเคียงเข้าด้วยกัน ถ้าโรย noise หลังหมุน noise จะคมกว่าความเป็นจริงเล็กน้อย
    (ของจริง noise เกิดตอนสแกน แล้วค่อยผ่านกระบวนการอื่น ไม่ใช่กลับกัน)
    """
    img = augment_intensity(img)
    img = augment_geometric(img)
    return img


def augment_shape(img: Image.Image) -> Image.Image:
    """
    เฉพาะ 3 ตัวที่หัวไฟล์ระบุว่าห้ามใช้ — flip / scale / elastic

    แยกไว้เป็นชุดเดี่ยวเพื่อวัดผลของมันล้วนๆ โดยไม่มีตัวอื่นปน ถ้าจะสรุปว่า
    "ข้อห้ามนี้ถูกหรือผิด" ต้องมีชุดนี้ ไม่ใช่ดูจาก standard_shape อย่างเดียว
    (ซึ่งมีทั้งของที่ช่วยและของที่อาจทำลายปนกันอยู่)
    """
    img = random_flip(img, p_h=0.5, p_v=0.5)
    img = random_scale(img, 0.85, 1.15, p=0.5)
    img = random_elastic(img, alpha=8.0, sigma=12.0, p=0.5)
    return img


def augment_standard_shape(img: Image.Image) -> Image.Image:
    """
    standard + 3 ตัวที่ห้ามไว้ (flip / scale / elastic)

    เรียง intensity -> geometric -> shape เพื่อให้ transform ที่ใช้ interpolation
    ทั้งหมด (rotate / scale / elastic) อยู่ติดกันท้ายสุด ภาพจึงผ่านการเกลี่ยพิกเซล
    เป็นชุดเดียว ไม่ใช่สลับกับการโรย noise ไปมาจนเนื้อภาพนุ่มกว่าที่ควรเป็น

    ⚠️ ชุดนี้เปลี่ยนรูปทรงซึ่งเป็นสิ่งที่ label อ้างอิงโดยตรง ต้องรายงานผลคู่กับ
    standard เสมอ ไม่งั้นแยกไม่ออกว่าผลที่เปลี่ยนมาจากการขยายข้อมูลหรือจากการบิดรูป
    """
    img = augment_intensity(img)
    img = augment_geometric(img)
    img = augment_shape(img)
    return img


def augment_strong_shape(img: Image.Image) -> Image.Image:
    """
    strong + 3 ตัวที่เปลี่ยนรูปทรง (flip / scale / elastic)

    เหตุผลที่ทำ: จาก sweep พบว่า strong ได้ macro F1 สูงสุด (0.614) ส่วน
    standard_shape ได้ mild/moderate สูงสุด (0.457 / 0.627) — เก่งคนละด้าน
    และการเติม shape ให้ standard เพิ่มผลได้ +3.5 pp (0.577 -> 0.612)
    ชุดนี้ทดสอบว่าเติมให้ strong แล้วได้ผลเหมือนกันไหม

    ⚠️ ข้อควรระวังที่ต่างจาก standard_shape: strong มีการหมุน/เลื่อนแรงกว่าอยู่แล้ว
    (±10°/±7% ที่ p=0.7 เทียบกับ ±7°/±5% ที่ p=0.5) การเติม scale/elastic/flip
    เข้าไปอีกจึงอาจถึงจุดอิ่มตัว หรือบิดจนภาพเสียมากกว่าได้ประโยชน์
    โอกาสที่ภาพจะไม่ถูกแตะเลยเหลือ 0.14% และเฉลี่ยโดน 4.2 transform ต่อภาพ
    (standard_shape อยู่ที่ 0.77% และ 3.6 ตัว)
    """
    img = augment_strong(img)
    img = augment_shape(img)
    return img


# ============================================================================
# ทะเบียนชื่อ -> ฟังก์ชัน (โครงเดียวกับ PREPROCESS_FNS ใน preprocessing.py)
# ============================================================================

AUGMENT_FNS = {
    "none": None,              # ไม่ทำอะไรเลย = พฤติกรรมเดิมของทุก config ที่มีอยู่
    "no_augment": no_augment,  # เดินผ่านปลั๊กแต่ไม่เปลี่ยนภาพ (ไว้ทดสอบว่าปลั๊กถูก)
    "intensity": augment_intensity,
    "geometric": augment_geometric,
    "standard": augment_standard,
    "strong": augment_strong,
    # --- กลุ่มที่เปลี่ยนรูปทรง (ดูคำเตือนที่หัวข้อฟังก์ชัน) ---
    "shape": augment_shape,
    "standard_shape": augment_standard_shape,
    "strong_shape": augment_strong_shape,
}

# ชุดที่มี transform เชิงเรขาคณิต — ใช้กับ resize_mode="stretch" ไม่ได้ เพราะไม่มี
# พื้นที่ดำรองรับ มุมกระดูกจะถูกตัดหายทันที เก็บไว้ที่เดียวเพื่อให้ train.py กับ
# สคริปต์อื่นเช็คตรงกัน ไม่ต้องไล่แก้หลายที่เวลาเพิ่มชุดใหม่
GEOMETRIC_AUGMENTS = {"geometric", "standard", "strong", "shape", "standard_shape",
                      "strong_shape"}


def get_augment_fn(name):
    """
    แปลง "ชื่อจาก config" (data.augment) เป็นฟังก์ชันจริง

    ไม่ระบุมา (None) = ไม่ทำ augmentation ซึ่งเป็นพฤติกรรมเดิมของทุก config
    ที่มีอยู่แล้ว จึงไม่ต้องแก้ config เก่าสักไฟล์
    """
    if name is None:
        return None

    key = str(name).strip().lower()
    if key not in AUGMENT_FNS:
        valid = ", ".join(sorted(AUGMENT_FNS))
        raise ValueError(
            f"data.augment = '{name}' ไม่รู้จัก — ต้องเป็นหนึ่งใน: {valid}\n"
            f"(ถ้าเพิ่งเขียนชุดใหม่ อย่าลืมลงทะเบียนใน AUGMENT_FNS ที่ augment.py)"
        )
    return AUGMENT_FNS[key]
