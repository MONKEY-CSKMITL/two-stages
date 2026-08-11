# Vertebral Fracture Classification (Stage 2) — วิธีรันโปรเจกต์

จำแนกระดับความรุนแรงของกระดูกสันหลังยุบ (Genant grade 0–3) จากภาพ DXA
ทีละปล้อง (T3–L5) เอกสารนี้ครอบคลุมตั้งแต่ติดตั้งจนถึงเทรนและดูผล

---

## 0. สิ่งที่ต้องมีก่อนเริ่ม

- Python 3.10 ขึ้นไป
- (แนะนำ) การ์ดจอ NVIDIA — เทรนบน CPU ได้แต่ช้ากว่ามาก (เป็นชั่วโมงต่อการทดลอง
  เทียบกับ 30–45 นาทีบน GPU)
- ข้อมูลดิบ 3 อย่าง (**ไม่รวมอยู่ใน git** — ต้องขอมาแยกต่างหาก เพราะเป็นข้อมูล
  ผู้ป่วยจริง): mask, ภาพ X-ray ต้นฉบับ, ไฟล์ DataTable.xlsx

---

## 1. ติดตั้งเครื่อง

### 1.1 สร้าง virtual environment

```powershell
cd two_stage_approach
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 1.2 ติดตั้ง library พื้นฐาน

```powershell
pip install numpy pandas pillow openpyxl scikit-learn iterative-stratification matplotlib
```

### 1.3 ติดตั้ง PyTorch

**ถ้ามีการ์ดจอ NVIDIA** — เช็ค CUDA version ก่อน:

```powershell
nvidia-smi
```

ดูเลขที่ขึ้นมุมขวาบนตาราง (`CUDA Version: xx.x`) แล้วติดตั้งตามนั้น เช่นถ้าขึ้น
12.x ขึ้นไป:

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

**ถ้าไม่มีการ์ดจอ NVIDIA** — ติดตั้งเวอร์ชัน CPU ธรรมดา:

```powershell
pip install torch
```

**ตรวจสอบว่าติดตั้งถูก**:

```powershell
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

### 1.4 ติดตั้ง timm (backbone ของโมเดล)

```powershell
pip install timm
```

### 1.5 (ไม่บังคับ) ถ้าจะลอง backbone RAD-DINO

```powershell
pip install transformers
```

ครั้งแรกที่รันจะดาวน์โหลด weight ~350MB จาก HuggingFace อัตโนมัติ

### 1.6 (ไม่บังคับ) ถ้าจะทำ Grad-CAM (ดูว่าโมเดลมองตรงไหนของภาพ)

```powershell
pip install grad-cam
```

---

## 2. วางข้อมูลดิบ

สร้างโครงสร้างนี้ไว้ที่ root ของโปรเจกต์:

```
data/raw/
├── masks/              ไฟล์ mask ทั้งหมด (XXXX_mask.png)
├── hologic/             ภาพ X-ray ต้นฉบับ (XXXX.png)
└── DataTable.xlsx       ไฟล์ Excel ที่มี grade + ข้อมูลผู้ป่วย
```

**เช็คว่าวางถูก**:

```powershell
dir data\raw\masks | measure
dir data\raw\hologic | measure
dir data\raw\DataTable.xlsx
```

---

## 3. รัน pipeline เตรียมข้อมูล (ทำครั้งเดียว)

รันตามลำดับนี้ แต่ละขั้นใช้ผลจากขั้นก่อนหน้า

### 3.1 ตัดปล้องกระดูกจาก mask + X-ray

```powershell
python .\scripts\crop.py
```

สร้างไฟล์รูป crop ไว้ที่ `data\interim\crops\` (แยกโฟลเดอร์ตามคนไข้) พร้อม
`manifest.csv` — ใช้เวลาสักครู่ ขึ้นกับจำนวนคนไข้

**เช็คผล**: `dir data\interim\crops` ควรเห็นโฟลเดอร์ย่อยหลายร้อยโฟลเดอร์

### 3.2 จับคู่ Genant grade จาก Excel เข้า manifest

```powershell
python .\scripts\adapt_manifest.py `
    --crops_dir data\interim\crops `
    --label_xlsx data\raw\DataTable.xlsx `
    --id_width 4 `
    --out_dir data\processed
```

**เช็คผล**: `dir data\processed\*.csv` ควรเห็น `manifest_xray_bbox.csv` และ
`manifest_xray_masked.csv`

### 3.3 แบ่งข้อมูล train / val / test

```powershell
python .\scripts\run_split.py `
    --manifests data\processed\manifest_xray_bbox.csv data\processed\manifest_xray_masked.csv `
    --out_dir data\processed\splits `
    --seed 42
```

**เช็คผล**: `dir data\processed\splits` ควรเห็นไฟล์ `xray_bbox_train.csv`,
`xray_bbox_val.csv`, `xray_bbox_test.csv` และแบบ `xray_masked_*` อีก 3 ไฟล์
พร้อม `split_summary.csv`

> ทำ 3 ขั้นนี้แค่ครั้งเดียว — หลังจากนี้ทุกการทดลองอ่านไฟล์ split ชุดเดียวกัน
> ไม่ต้องรันซ้ำ (ยกเว้นจะเปลี่ยนข้อมูลดิบ)

---

## 4. เทรนโมเดล

### 4.1 โครงสร้างไฟล์ config

การทดลองแต่ละแบบควบคุมด้วยไฟล์ `.yaml` ในโฟลเดอร์ `configs\` — เปลี่ยน
การทดลองโดยแก้ไฟล์ config ไม่ต้องแก้โค้ด

ตัวอย่างโครง:

```yaml
experiment:
  name: effb0_multiclass_masked   # ชื่อการทดลอง (ใช้ตั้งชื่อโฟลเดอร์ผลลัพธ์)
  seed: 42

data:
  split_dir: data/processed/splits
  variant: xray_masked            # หรือ xray_bbox
  task: multiclass                # หรือ binary
  img_size: null                  # null = ใช้ขนาดมาตรฐานของ backbone อัตโนมัติ

model:
  backbone: efficientnet_b0       # หรือ convnext_tiny, rad_dino
  pretrained: true
  level_embed_dim: 32
  head_dropout: 0.2
  freeze_backbone: false          # true แนะนำถ้าใช้ rad_dino

loss:
  gamma: 2.0
  use_class_weights: true

train:
  epochs: 15
  batch_size: 64                  # ลดถ้าเจอ CUDA out of memory
  lr: 1.0e-4
  weight_decay: 1.0e-4
  patience: 3                     # หยุดก่อนถ้า val ไม่ดีขึ้นติดกันเท่านี้ epoch
  num_workers: 4                  # เปลี่ยนเป็น 0 ถ้าเจอปัญหาค้างบน Windows

output:
  dir: outputs/runs/effb0_multiclass_masked
```

### 4.2 รันเทรน

```powershell
python .\scripts\train.py --config configs\stage2_efficientnet_b0.yaml
```

ระหว่างเทรนจะเห็น log แบบนี้ทุก epoch:

```
epoch 01  train_loss=1.20  val_loss=1.59  val_F1=0.24  val_AUC=0.53
```

รันจบจะพิมพ์สรุปผลบนชุด test พร้อม classification report และ confusion matrix

### 4.3 ผลลัพธ์ที่ได้

```
outputs/runs/{ชื่อการทดลอง}/
├── best.pt                              weight ของโมเดลที่ดีที่สุด (เลือกจาก val F1)
├── metrics.json                         ผลบนชุด test + recall แยกราย grade
├── config_used.yaml                     สำเนา config ที่ใช้จริง
├── history.csv                          ค่า loss/metric ราย epoch
├── plots/
│   ├── loss_curve.png                   train loss vs val loss
│   ├── metric_curve.png                 val F1 และ val AUC ราย epoch
│   ├── confusion_matrix.png             ตารางสับสนรวมทุกปล้อง
│   └── confusion_matrix_per_level.png   ตารางสับสนแยก 15 ปล้อง
└── tables/
    ├── metrics_overall.csv              precision/recall/f1 รวมทุกปล้อง
    └── metrics_per_level.csv            precision/recall/f1 แยกรายปล้อง
```

---

## 5. ทดลองหลายแบบ (สลับ backbone / crop / โจทย์)

สร้างไฟล์ config ใหม่โดยก๊อปไฟล์เดิมแล้วแก้บางบรรทัด **เปลี่ยนทีละอย่าง**
เพื่อให้เทียบผลได้ตรง (ตัวแปรอื่นต้องเหมือนเดิมทุกบรรทัด)

| อยากลอง | แก้อะไรใน config |
|---|---|
| backbone อื่น | `model.backbone` |
| ใช้ crop แบบไม่มีพื้นหลัง | `data.variant: xray_masked` |
| โจทย์ 2 คลาส (เสียหาย/ไม่เสียหาย) | `data.task: binary` |
| เพิ่มข้อมูลผู้ป่วย (อายุ/เพศ/น้ำหนัก/ส่วนสูง) | เพิ่ม `data.metadata_xlsx: data/raw/DataTable.xlsx` และ `model.use_metadata: true` |
| ยืดภาพเต็มกรอบแทนการเติมขอบดำ | `data.resize_mode: stretch` (ไม่ใส่ = `pad`) |
| แต่งภาพก่อนป้อนโมเดล | `data.preprocess: ชื่อฟังก์ชัน` — ชื่อที่ใช้ได้ดูจาก `PREPROCESS_FNS` ใน `src/core/stage2/preprocessing.py` (ไม่ใส่ = ไม่แต่งภาพ) |

**อย่าลืมเปลี่ยน `experiment.name` และ `output.dir`** ทุกครั้งที่สร้าง config
ใหม่ ไม่งั้นผลจะเขียนทับของเดิม

---

## 6. ดูว่าโมเดลมองตรงไหนของภาพ (Grad-CAM)

ใช้ `best.pt` ที่เทรนไว้แล้ว ไม่ต้องเทรนใหม่ ใช้เวลา 1–2 นาที:

```powershell
python .\scripts\gradcam.py --config configs\stage2_effb0_masked.yaml
```

ได้ภาพที่ `outputs\runs\{ชื่อ}\plots\gradcam_test.png` — สีแดง/เหลือง =
บริเวณที่มีอิทธิพลต่อคำตอบมาก, กรอบเขียว/แดง = ทายถูก/ผิด

> ใช้ได้เฉพาะ backbone ตระกูล CNN (EfficientNet, ConvNeXt) — RAD-DINO เป็น
> Vision Transformer ยังไม่รองรับ

---

## 7. ปัญหาที่เจอบ่อย

| อาการ | สาเหตุ / วิธีแก้ |
|---|---|
| `UnicodeDecodeError` ตอนอ่าน config | เปิดไฟล์ .yaml ด้วย VS Code เช็คมุมขวาล่างว่าเป็น UTF-8 ไหม ถ้าไม่ใช่ให้ save ใหม่เป็น UTF-8 |
| `CUDA out of memory` | ลด `train.batch_size` ใน config (64 → 32 → 16) |
| ค้าง/error ตอนเทรน (Windows) | ลด `train.num_workers` เป็น 0 |
| `device: cpu` ทั้งที่มีการ์ดจอ | เช็คว่าติดตั้ง torch เวอร์ชัน CUDA ถูกไหม (ขั้นตอน 1.3) |
| error `pip's dependency resolver...torchvision` | เป็นแค่คำเตือน ไม่กระทบ (โปรเจกต์นี้ไม่ได้ใช้ torchvision เลย) |
| อยากใช้ RAD-DINO แต่ VRAM ไม่พอ | ตั้ง `img_size: 224` (ไม่ใช่ 518 ตามสเปกทางการ) และ `freeze_backbone: true` ใน config |

---

## 8. โครงสร้างโปรเจกต์โดยย่อ

```
configs/           ไฟล์ตั้งค่าการทดลอง (.yaml)
data/
  raw/              ข้อมูลดิบ (ไม่อยู่ใน git — ต้องขอมาเอง)
  interim/          ผล crop (สร้างจากขั้น 3.1)
  processed/        manifest + split (สร้างจากขั้น 3.2–3.3)
outputs/runs/       ผลการเทรนแต่ละการทดลอง
scripts/            ไฟล์ที่รันตรงๆ: crop.py, adapt_manifest.py, run_split.py,
                    train.py, gradcam.py
src/core/
  stage2/           โค้ดหลัก: dataset.py, transforms.py, model.py, losses.py,
                    split.py, preprocessing.py
  utils/            reporting.py (สร้างกราฟ/ตาราง)
notebooks/          เอกสารสรุปผลการทดลอง (experiment_summary.md)
```

ไล่ลำดับการทำงาน: `crop.py → adapt_manifest.py → run_split.py → train.py
(→ gradcam.py)`