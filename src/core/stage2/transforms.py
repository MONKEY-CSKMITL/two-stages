"""
transforms.py — เตรียมรูป crop ให้พร้อมป้อนโมเดล (ไฟล์นี้เป็น "ห้องสมุด" ไม่ได้รันเอง)

รวมวิธีเตรียมรูปไว้ในไฟล์เดียว สลับด้วยชื่อ backbone ตัวเดียว

ทำไมข้างในต้องมี 2 เส้นทาง (แม้จะรวมไว้ไฟล์เดียว):
  - EfficientNet, ConvNeXt ฝึกมาด้วยค่า normalize แบบ ImageNet
  - RAD-DINO ฝึกมาด้วยค่า normalize แบบ MIMIC-CXR (คนละชุดกัน)

ทั้ง 2 เส้นทางเขียนเองด้วย PIL+numpy ตรงๆ ไม่พึ่ง library เสริม (ไม่ต้องมี
transformers) — พารามิเตอร์ของเส้นทาง RAD-DINO (ค่า resize/crop/mean/std) มาจาก
ไฟล์ preprocessor_config.json ทางการของ microsoft/rad-dino บน HuggingFace
(ตรวจสอบแล้วว่าเป็นขั้นตอนตายตัว 4 ขั้น ไม่ใช่ black box) เพราะเขียนเองได้ทั้งคู่
พารามิเตอร์ preprocess_fn จึงใช้ได้กับทั้ง 2 เส้นทางเท่ากัน ไม่ต้องกันไว้อีกต่อไป

ข้อควรรู้: ค่าพารามิเตอร์ RAD-DINO ด้านล่าง copy มาจาก config ที่เผยแพร่ ณ ตอนที่
ตรวจสอบ — ถ้า Microsoft อัปเดต config ทีหลัง ค่าที่นี่จะไม่รู้ตัวและไม่อัปเดตตาม
(ต่างจากการเรียก AutoImageProcessor.from_pretrained ที่จะได้ค่าล่าสุดเสมอ)
เหมาะกับงานที่ต้องการความโปร่งใส/ทดสอบซ้ำได้แน่นอน มากกว่าความทันสมัยอัตโนมัติ
"""

# --- import library ที่ต้องใช้ (ทั้ง 2 เส้นทาง ใช้ชุดเดียวกันหมด) ---
import numpy as np           # จัดการตารางตัวเลขของรูปภาพ
from PIL import Image        # เปิด/ปรับขนาดรูปภาพ


# ค่าเฉลี่ยและส่วนเบี่ยงเบนมาตรฐานของชุดข้อมูล ImageNet — ใช้กับ EfficientNet/ConvNeXt
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# ค่าที่ใช้กับ RAD-DINO — คัดลอกมาจาก preprocessor_config.json ทางการ
# (microsoft/rad-dino, commit 364cf91) ทั้ง 3 ช่องสีใช้ค่าเดียวกัน เพราะภาพต้นทาง
# เป็น X-ray ขาวดำที่ทำเป็น 3 ช่องซ้อนกัน ไม่ใช่ภาพสีจริง
RAD_DINO_MEAN = np.array([0.5307, 0.5307, 0.5307], dtype=np.float32)
RAD_DINO_STD = np.array([0.2583, 0.2583, 0.2583], dtype=np.float32)
RAD_DINO_SIZE = 518   # official config: shortest_edge=518, center crop 518x518

RAD_DINO_BACKBONES = {"rad_dino"}


def prepare_image(path: str, backbone: str = "efficientnet_b0", size: int = None,
                  preprocess_fn=None, resize_mode: str = "pad") -> np.ndarray:
    """
    จุดเข้าเดียวที่ dataset.py เรียกใช้ — เลือกวิธีเตรียมรูปให้อัตโนมัติตามชื่อ backbone

    input:  path         = ที่อยู่ไฟล์รูป
            backbone     = ชื่อ backbone ที่จะเอาไปเทรน (ใช้ตัดสินว่าต้อง normalize แบบไหน)
            size         = ขนาดรูปที่ต้องการ — ไม่ใส่ (None, ค่าเริ่มต้น) จะได้ขนาดมาตรฐาน
                           ของ backbone นั้นอัตโนมัติ: 224 สำหรับ backbone ทั่วไป, 518 สำหรับ
                           rad_dino (ตามสเปกทางการ) — ใส่เลขมาเองถ้าต้องการบังคับขนาดอื่น
                           เช่น ตั้งใจเทียบทุก backbone ที่ 224 เท่ากันเพื่อความแฟร์
            preprocess_fn = (ไม่บังคับ) ฟังก์ชันเสริมสำหรับแต่งภาพเพิ่ม เช่น CLAHE, denoise
                            ถ้าไม่ใส่ (ค่าเริ่มต้น None) พฤติกรรมเหมือนเดิมทุกประการ
                            ใช้ได้ทั้ง 2 เส้นทางเท่ากัน (ทั้งคู่เขียนเองด้วย PIL แล้ว)
            resize_mode  = "pad" (ค่าเริ่มต้น) เติมขอบดำ รักษาสัดส่วนกระดูก
                           "stretch" ยืดเต็มกรอบ ไม่รักษาสัดส่วน — ใช้ได้เฉพาะเส้นทาง
                           backbone ทั่วไปเท่านั้น (rad_dino ใช้ center-crop ตามสเปกทางการ
                           อยู่แล้ว ไม่มีพื้นที่ว่างให้เลือกวิธีเติม จึงไม่รับพารามิเตอร์นี้)
    output: numpy array รูปร่าง (3, H, W) พร้อมป้อนโมเดล
    """
    if backbone in RAD_DINO_BACKBONES:
        # ไม่ใส่ size มา -> ใช้ 518 ตามสเปกทางการของ RAD-DINO
        actual_size = size if size is not None else RAD_DINO_SIZE
        return _prepare_rad_dino(path, actual_size, preprocess_fn)
    # ไม่ใส่ size มา -> ใช้ 224 (ค่ามาตรฐานของ backbone ตระกูล timm ทั่วไป)
    actual_size = size if size is not None else 224
    return _prepare_standard(path, actual_size, preprocess_fn, resize_mode)


def _prepare_standard(path: str, size: int, preprocess_fn=None,
                      resize_mode: str = "pad") -> np.ndarray:
    """
    เส้นทางทั่วไป: ย่อ+3ช่องสี+normalize แบบ ImageNet
    ใช้กับ backbone ตระกูล timm ทั่วไป (EfficientNet, ConvNeXt, ฯลฯ)

    resize_mode ควบคุมวิธีทำให้รูปเป็นจัตุรัส (มี 2 แบบ แยกจากกันเด็ดขาด):
      "pad"    (ค่าเริ่มต้น — พฤติกรรมเดิมของโค้ด ไม่เปลี่ยนอะไรถ้าไม่ระบุ)
               ย่อรูปให้พอดีด้านยาวสุด รักษาสัดส่วนเดิม แล้วเติมขอบดำรอบๆ
               ข้อดี: กระดูกไม่ผิดสัดส่วนเลย
               ข้อเสีย: มีพื้นที่ดำว่างเปล่า ซึ่ง Grad-CAM เคยพบว่าโมเดลบางเคส
               ไปมองพื้นที่ว่างนี้แทนตัวกระดูกจริง

      "stretch" (ตัวเลือกใหม่ — ต้องระบุมาเท่านั้นถึงจะใช้)
               ยืด/บีบรูปให้เต็มจัตุรัสตรงๆ ไม่รักษาสัดส่วน ไม่มีพื้นที่ว่างเลย
               ข้อดี: ไม่มีพื้นที่ดำให้โมเดลหลบไปมอง
               ข้อเสีย: กระดูกผิดสัดส่วน — แต่ละปล้องโดนยืดไม่เท่ากัน (ปล้องที่แบน
               จากการหักรุนแรงจะโดนยืดแนวตั้งมากกว่าปล้องปกติ) อาจกระทบการวัด
               ความสูงซึ่งเป็นหลักการวัด Genant grade โดยตรง — ต้องตรวจด้วย
               Grad-CAM ควบคู่เสมอว่าโมเดลกลับไปมองกระดูกจริงหรือยังมองที่อื่น
    """
    img = Image.open(path).convert("L")

    if preprocess_fn is not None:
        img = preprocess_fn(img)

    if resize_mode == "stretch":
        # ยืดตรงๆ ให้เป็น size x size โดยไม่สนสัดส่วนเดิมเลย
        # (ไม่ต้องคำนวณ canvas/paste เพราะไม่มีพื้นที่ว่างเหลือให้เติม)
        img = img.resize((size, size))

    elif resize_mode == "pad":
        # ย่อรูปให้พอดีกรอบ โดยรักษาสัดส่วนเดิม (ไม่บีบ ไม่ยืด กันกระดูกผิดสัดส่วน)
        w, h = img.size
        scale = size / max(w, h)
        new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
        img = img.resize((new_w, new_h))

        # เติมขอบดำให้เป็นจัตุรัสขนาด size x size (วางรูปไว้กึ่งกลาง)
        canvas = Image.new("L", (size, size), 0)
        canvas.paste(img, ((size - new_w) // 2, (size - new_h) // 2))
        img = canvas

    else:
        raise ValueError(f"resize_mode ต้องเป็น 'pad' หรือ 'stretch' เท่านั้น ได้รับ '{resize_mode}'")

    gray = np.array(img, dtype=np.float32) / 255.0
    rgb = np.stack([gray, gray, gray], axis=-1)
    rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    return rgb.transpose(2, 0, 1).astype(np.float32)


def _prepare_rad_dino(path: str, size: int, preprocess_fn=None) -> np.ndarray:
    """
    เส้นทาง RAD-DINO — เขียนเองตามสูตรจาก preprocessor_config.json ทางการ 4 ขั้นตอน:
      1. resize ให้ด้านสั้นสุด = size พิกเซล (รักษาสัดส่วน, bicubic ตาม resample=3 ในไฟล์ config)
      2. ตัดกึ่งกลางให้เหลือ size x size (center crop — ต่างจากเส้นทางทั่วไปที่ "เติมขอบดำ"
         เส้นทางนี้ "ตัดพื้นที่ส่วนเกิน" ทิ้งแทน ตามที่ config กำหนดไว้)
      3. หารด้วย 255 (rescale_factor)
      4. normalize ด้วยค่าเฉลี่ย/ส่วนเบี่ยงเบนมาตรฐานของ RAD-DINO (ไม่ใช่ ImageNet)

    size ควรเป็น 518 (ค่าทางการ) ถ้าจะบังคับขนาดอื่น (เช่น 224 เพื่อเทียบกับ backbone อื่น
    แบบ input size เท่ากัน) ทำได้เพราะ ViT ของ RAD-DINO ใช้ patch size 14 ซึ่ง 224 หารลงตัว
    (224/14=16) แต่เป็นการเบี่ยงเบนจากสเปกทางการ — ควรระบุในเล่มว่าทำแบบนี้ด้วยเหตุผลอะไร
    """
    img = Image.open(path).convert("RGB")   # do_convert_rgb: true ใน config

    if preprocess_fn is not None:
        img = preprocess_fn(img)

    # ขั้นที่ 1: resize ให้ด้านสั้นสุด = size (ต่างจากเส้นทางทั่วไปที่ใช้ด้านยาวสุด)
    w, h = img.size
    scale = size / min(w, h)
    new_w, new_h = max(size, round(w * scale)), max(size, round(h * scale))
    img = img.resize((new_w, new_h), resample=Image.BICUBIC)

    # ขั้นที่ 2: ตัดกึ่งกลางให้เหลือ size x size พอดี
    left = (new_w - size) // 2
    top = (new_h - size) // 2
    img = img.crop((left, top, left + size, top + size))

    # ขั้นที่ 3-4: แปลงเป็นตัวเลข, หารด้วย 255, normalize ด้วยค่าเฉพาะของ RAD-DINO
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - RAD_DINO_MEAN) / RAD_DINO_STD

    return arr.transpose(2, 0, 1).astype(np.float32)