"""
model.py — โครงสร้างโมเดล Stage 2 (ไฟล์นี้เป็น "ห้องสมุด" ไม่ได้รันเอง)

โมเดลรับ input 2 อย่าง แล้วคายคำตอบ 1 อย่าง:

    รูป crop ─────→ [backbone] ────→ feature (768-1280 ตัวเลข) ┐
                                                                ├→ [head] → grade
    ปล้องที่เท่าไหร่ → [embedding] → เวกเตอร์ (32 ตัวเลข)      ┘

ทำไมต้องบอก "ปล้องไหน" ให้โมเดลด้วย:
  กระดูกบางปล้อง (โดยเฉพาะช่วงอก) มีรูปทรงเอียงเล็กน้อยตามธรรมชาติ ซึ่งหน้าตาคล้าย
  การหักระดับเล็กน้อย ถ้าโมเดลไม่รู้ว่ากำลังดูปล้องไหน อาจสับสนระหว่าง "ปกติของปล้องนี้"
  กับ "หักจริง" — การบอกปล้องเข้าไปช่วยให้แยกแยะได้ดีขึ้น โดยยังใช้ backbone ตัวเดียว
  ร่วมกันทุกปล้อง (ได้ประโยชน์จากการเรียนรู้ร่วมกัน ไม่ต้องแยกโมเดล 15 ตัว)

รองรับ backbone 2 ตระกูล สลับด้วยชื่อเดียว:
  - ตระกูล timm ทั่วไป: "efficientnet_b0", "convnext_tiny" ฯลฯ (ต้องมี pip install timm)
  - "rad_dino": microsoft/rad-dino จาก HuggingFace (ต้องมี pip install transformers)
    หมายเหตุ: ส่วน "เตรียมรูป" ของ rad_dino เราเขียนเองใน transforms.py แล้ว ไม่ต้องพึ่ง
    transformers — แต่ส่วน "ตัวโมเดล" (weight ที่ฝึกมาแล้ว) ยังต้องโหลดจาก HuggingFace อยู่
"""

# --- import library ที่ต้องใช้ ---
import torch                    # ตัวหลักของ PyTorch
import torch.nn as nn           # nn = neural network โมดูลที่มีชิ้นส่วนโมเดลสำเร็จรูป

NUM_LEVELS = 15                 # จำนวนปล้องกระดูกทั้งหมด (T3 ถึง L5)
RAD_DINO_HF_NAME = "microsoft/rad-dino"   # ชื่อโมเดลบน HuggingFace
RAD_DINO_FEATURE_DIM = 768      # ขนาด feature ที่ RAD-DINO คายออกมา (ViT-B hidden size)


class RadDinoBackbone(nn.Module):
    """
    ตัวห่อ (wrapper) ให้ RAD-DINO มีหน้าตาเหมือน backbone จาก timm

    ทำไมต้องห่อ: โค้ดข้างล่างคาดหวังว่า backbone ทุกตัวจะมี
      - attribute ชื่อ .num_features บอกขนาด feature
      - เรียก backbone(x) แล้วได้ feature กลับมาเป็น (จำนวนรูป, ขนาด feature)
    แต่ RAD-DINO จาก HuggingFace มีหน้าตาต่างออกไป (คืน object ที่มีหลายอย่างข้างใน)
    เลยต้องห่อให้เข้ารูปเดียวกัน โค้ดส่วนอื่นจะได้ไม่ต้องรู้ว่าใช้ backbone ตัวไหนอยู่
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()   # เรียก __init__ ของ nn.Module (คลาสแม่) ก่อนเสมอ

        # import ตรงนี้ (ไม่ใช่บนสุดของไฟล์) เพื่อไม่บังคับให้คนที่ใช้แค่ EfficientNet
        # ต้องติดตั้ง transformers ด้วย — จะโหลดก็ต่อเมื่อเลือกใช้ rad_dino จริงๆ
        try:
            from transformers import AutoModel
        except ImportError:
            raise ImportError(
                "backbone='rad_dino' ต้องติดตั้ง transformers ก่อน: pip install transformers"
            )

        if not pretrained:
            # RAD-DINO มีค่าเฉพาะตรงที่ "ฝึกมาแล้วบนภาพ X-ray" ถ้าไม่เอา weight ที่ฝึกมา
            # ก็ไม่มีเหตุผลจะใช้มันแทน backbone อื่นเลย
            raise ValueError("backbone='rad_dino' ต้องใช้ pretrained=True เท่านั้น")

        self.model = AutoModel.from_pretrained(RAD_DINO_HF_NAME)
        self.num_features = RAD_DINO_FEATURE_DIM   # ให้หน้าตาเหมือน timm backbone

    def forward(self, x):
        """
        x = รูปทั้ง batch รูปร่าง (จำนวนรูป, 3, สูง, กว้าง)
        คืน feature รูปร่าง (จำนวนรูป, 768)
        """
        outputs = self.model(pixel_values=x)
        # pooler_output = "บทสรุปของทั้งรูป" ที่ ViT คายออกมา (เรียกว่า CLS token)
        # เป็นเวกเตอร์เดียวต่อ 1 รูป เหมาะกับงานจำแนกประเภทพอดี
        return outputs.pooler_output


class VertebraClassifier(nn.Module):
    """
    โมเดลหลัก — ประกอบจาก 3 ชิ้น: backbone + level embedding + head
    """

    def __init__(self,
                 backbone: str = "efficientnet_b0",
                 num_classes: int = 4,
                 num_levels: int = NUM_LEVELS,
                 level_embed_dim: int = 32,
                 pretrained: bool = True,
                 head_dropout: float = 0.2,
                 freeze_backbone: bool = False,
                 use_metadata: bool = False,
                 num_metadata: int = 4,
                 metadata_embed_dim: int = 16):
        """
        backbone        = ชื่อ backbone ("efficientnet_b0", "convnext_tiny", "rad_dino", ...)
        num_classes     = จำนวนคำตอบที่เป็นไปได้ (4 สำหรับ multiclass, 2 สำหรับ binary)
        num_levels      = จำนวนปล้อง (15 เสมอ)
        level_embed_dim = ขนาดเวกเตอร์ที่ใช้แทนแต่ละปล้อง (32 = ค่ากลางๆ ที่เลือกมา)
        pretrained      = ใช้ weight ที่ฝึกมาแล้วไหม (ควรเป็น True เสมอ เพราะข้อมูลเราน้อย)
        head_dropout    = สัดส่วนที่สุ่มปิดตอนเทรน กัน overfit (0.2 = ปิด 20%)
        freeze_backbone = ล็อก backbone ไม่ให้เรียนรู้เพิ่มไหม (เหมาะกับ backbone ใหญ่ + ข้อมูลน้อย)
        use_metadata    = ใช้ข้อมูลผู้ป่วย (อายุ เพศ น้ำหนัก ส่วนสูง) ร่วมด้วยไหม
                          ค่าเริ่มต้น False = ไม่ใช้ (พฤติกรรมเหมือนเดิมทุกประการ)
        num_metadata    = จำนวน feature ของ metadata (4: อายุ เพศ น้ำหนัก ส่วนสูง)
        metadata_embed_dim = ขนาดเวกเตอร์หลังแปลง metadata (16 = เล็กกว่า level embedding
                          เพราะข้อมูลน้อยกว่าและเป็นระดับคนไข้ ไม่ควรมีอิทธิพลมากเกินไป)
        """
        super().__init__()

        # --- ชิ้นที่ 1: backbone (ตัวดูรูปแล้วสกัดความหมาย) ---
        if backbone == "rad_dino":
            self.backbone = RadDinoBackbone(pretrained=pretrained)
        else:
            try:
                import timm
            except ImportError:
                raise ImportError(f"backbone='{backbone}' ต้องติดตั้ง timm ก่อน: pip install timm")
            # num_classes=0 = บอก timm ว่า "ไม่ต้องใส่หัวจำแนกมาให้ ขอแค่ feature ดิบ"
            # เพราะเราจะใส่หัวเองที่รวม level embedding เข้าไปด้วย
            self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)

        # .num_features = ขนาด feature ที่ backbone คายออกมา (แต่ละตัวไม่เท่ากัน
        # เช่น EfficientNet-B0 = 1280, ConvNeXt-Tiny = 768, RAD-DINO = 768)
        # อ่านค่าอัตโนมัติแบบนี้ ทำให้สลับ backbone ได้โดยไม่ต้องแก้เลขเอง
        feat_dim = self.backbone.num_features

        # ถ้าสั่งล็อก: ปิดการเรียนรู้ของทุกพารามิเตอร์ใน backbone
        # requires_grad=False แปลว่า "ตอน backprop ไม่ต้องคำนวณ/ปรับค่านี้"
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # --- ชิ้นที่ 2: level embedding (ตารางแปลง "ปล้องไหน" เป็นเวกเตอร์) ---
        # nn.Embedding(15, 32) = ตาราง 15 แถว x 32 คอลัมน์ ค่าเริ่มต้นสุ่ม
        # ค่าในตารางนี้เป็น weight ที่โมเดลเรียนรู้เองระหว่างเทรน (เหมือน weight ส่วนอื่น)
        self.level_embed = nn.Embedding(num_levels, level_embed_dim)

        # --- ชิ้นที่ 3 (ไม่บังคับ): metadata encoder ---
        # แปลงตัวเลข 4 ค่า (อายุ เพศ น้ำหนัก ส่วนสูง) เป็นเวกเตอร์ที่โมเดลใช้ได้
        # ใช้ Linear + ReLU ชั้นเดียว เพื่อให้โมเดลเรียนรู้ความสัมพันธ์ระหว่าง feature ได้บ้าง
        # (เช่น "อายุมาก + น้ำหนักน้อย" อาจมีความหมายต่างจากแต่ละอย่างแยกกัน)
        self.use_metadata = use_metadata
        if use_metadata:
            self.metadata_encoder = nn.Sequential(
                nn.Linear(num_metadata, metadata_embed_dim),
                nn.ReLU(),
            )
            extra_dim = metadata_embed_dim
        else:
            self.metadata_encoder = None
            extra_dim = 0   # ไม่ใช้ metadata -> ขนาด input ของ head เท่าเดิมทุกประการ

        # --- ชิ้นที่ 4: head (ตัวตัดสินใจตอบ) ---
        # nn.Sequential = ต่อชิ้นส่วนเรียงกัน ข้อมูลไหลผ่านทีละชิ้นตามลำดับ
        self.head = nn.Sequential(
            # Dropout = ตอนเทรน สุ่มปิดสัญญาณบางเส้นทาง กันโมเดลพึ่งพาเส้นทางใดเส้นทางหนึ่ง
            # มากเกินไปจนจำข้อมูลแทนที่จะเข้าใจ (ตอนวัดผล/ใช้จริง PyTorch ปิดฟังก์ชันนี้ให้เอง)
            nn.Dropout(head_dropout),
            # Linear = ชั้นคำนวณเชิงเส้น: รับ (feat_dim + level + metadata) ตัวเลข คาย num_classes ตัวเลข
            nn.Linear(feat_dim + level_embed_dim + extra_dim, num_classes),
        )

    def forward(self, image, level_idx, metadata=None):
        """
        image     = รูปทั้ง batch รูปร่าง (จำนวนรูป, 3, สูง, กว้าง)
        level_idx = ปล้องของแต่ละรูป รูปร่าง (จำนวนรูป,) ค่า 0-14
        metadata  = ข้อมูลผู้ป่วย รูปร่าง (จำนวนรูป, 4) — ใช้เฉพาะเมื่อ use_metadata=True
                    ถ้า use_metadata=False จะถูกเพิกเฉยทั้งหมด (ส่งมาหรือไม่ส่งก็ได้)

        คืน logits รูปร่าง (จำนวนรูป, num_classes)
        (logits = คะแนนดิบของแต่ละคำตอบ ยังไม่ได้แปลงเป็นความน่าจะเป็น)
        """
        img_feat = self.backbone(image)          # (N, feat_dim)  ← ความหมายที่สกัดจากรูป
        lvl_feat = self.level_embed(level_idx)   # (N, 32)         ← เวกเตอร์ประจำปล้องนั้น

        parts = [img_feat, lvl_feat]

        if self.use_metadata:
            if metadata is None:
                raise ValueError("โมเดลตั้งค่า use_metadata=True แต่ไม่ได้ส่ง metadata เข้ามา")
            parts.append(self.metadata_encoder(metadata))   # (N, metadata_embed_dim)

        # torch.cat(..., dim=1) = ต่อเวกเตอร์เข้าด้วยกันตามแนวนอน
        # เช่น 1280 + 32 (+16 ถ้าใช้ metadata) = 1312 หรือ 1328 ตัว (ยังเป็น 1 แถวต่อ 1 รูป)
        fused = torch.cat(parts, dim=1)

        return self.head(fused)


def build_model(backbone: str = "efficientnet_b0",
                num_classes: int = 4,
                num_levels: int = NUM_LEVELS,
                level_embed_dim: int = 32,
                pretrained: bool = True,
                head_dropout: float = 0.2,
                freeze_backbone: bool = False,
                use_metadata: bool = False,
                num_metadata: int = 4,
                metadata_embed_dim: int = 16) -> nn.Module:
    """
    ฟังก์ชันสร้างโมเดล — มีไว้เพื่อให้ train.py เรียกใช้ง่ายๆ ไม่ต้องรู้จักชื่อคลาสข้างใน
    (ถ้าวันหนึ่งเปลี่ยนโครงสร้างคลาสข้างใน train.py ก็ไม่ต้องแก้ตาม)
    """
    return VertebraClassifier(backbone, num_classes, num_levels, level_embed_dim,
                              pretrained, head_dropout, freeze_backbone,
                              use_metadata, num_metadata, metadata_embed_dim)


def count_parameters(model: nn.Module) -> dict:
    """
    นับจำนวนพารามิเตอร์ (ตัวเลขที่ปรับได้) ในโมเดล — ไว้พิมพ์ดูตอนเริ่มเทรน

    ประโยชน์: เห็นได้ทันทีว่า freeze_backbone ทำงานจริงไหม (ถ้าล็อกแล้ว trainable
    ควรลดลงเหลือแค่หลักหมื่น จากหลักล้าน) และเทียบขนาด backbone แต่ละตัวได้
    """
    total = sum(p.numel() for p in model.parameters())               # numel = จำนวนตัวเลขใน tensor นั้น
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}