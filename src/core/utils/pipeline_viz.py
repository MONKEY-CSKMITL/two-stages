"""
pipeline_viz.py — วาดภาพ "ทุกขั้นตอน" ที่ภาพหนึ่งใบเดินผ่านก่อนถึงโมเดล

แยกจาก reporting.py เพราะคนละหน้าที่: reporting.py รายงาน "ผลลัพธ์หลังเทรน"
ส่วนไฟล์นี้รายงาน "สิ่งที่ป้อนเข้าไปตอนเทรน" ซึ่งตรวจสอบได้ก่อนเทรนด้วยซ้ำ

ทำไมต้องมี:
  ท่อเตรียมภาพมีหลายขั้นซ้อนกัน (preprocess -> resize -> augment -> normalize)
  แต่ละขั้นแก้ภาพคนละแบบ ถ้าดูแค่ตัวเลขสรุปจะไม่มีทางรู้ว่าขั้นไหนทำอะไรพัง
  ตัวอย่างจริงที่เจอมาแล้วในโปรเจกต์นี้ 2 ครั้ง และทั้ง 2 ครั้งเห็นจากภาพเท่านั้น
  ตัวเลขไม่ฟ้อง:
    - CLAHE บนภาพ masked สร้าง artifact ที่รอยต่อกระดูกกับพื้นดำ
    - augmentation แบบแรงทำภาพที่สว่างอยู่แล้วขาวจนหมดรายละเอียด (ชนเพดาน 255)

หมายเหตุเรื่องภาษา: ข้อความในกราฟใช้ภาษาอังกฤษ ตามธรรมเนียมเดียวกับ reporting.py

ผลลัพธ์ที่สร้าง (ทั้ง 3 อย่างใช้ prefix เดียวกัน):
    {prefix}_stages.png      ตารางภาพ: แถว = ตัวอย่าง, คอลัมน์ = ขั้นตอน
    {prefix}_histograms.png  การกระจายค่าความสว่างของพิกเซลกระดูก แยกตามขั้นตอน
    {prefix}_stats.csv       ตัวเลขสรุปรายขั้นตอน (mean/sd/ช่วง/สัดส่วนที่อิ่มตัว)
"""

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core.stage2.transforms import prepare_image, IMAGENET_MEAN, IMAGENET_STD

GRADE_NAMES = {0: "normal", 1: "mild", 2: "moderate", 3: "severe"}


# ============================================================================
# ส่วนที่ 1: เดินท่อทีละขั้น แล้วเก็บภาพไว้ทุกขั้น
# ============================================================================

def _resize_like_transforms(img: Image.Image, size: int, resize_mode: str) -> Image.Image:
    """
    ทำ resize แบบเดียวกับ _prepare_standard ใน transforms.py เป๊ะ

    ต้องเขียนซ้ำเพราะ transforms.py คืนเฉพาะผลสุดท้าย ไม่ได้คืนภาพระหว่างทาง
    ความเสี่ยงคือถ้าวันหนึ่งมีคนแก้ transforms.py แล้วลืมแก้ตรงนี้ ภาพที่วาดจะโกหก
    — จึงมีการตรวจสอบอัตโนมัติใน collect_stages() ว่าผลสุดท้ายของสองเส้นทาง
    ต้องตรงกันเป๊ะ ถ้าไม่ตรงจะฟ้องทันที (ดู verify ในฟังก์ชันนั้น)
    """
    if resize_mode == "stretch":
        return img.resize((size, size))

    w, h = img.size
    scale = size / max(w, h)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    small = img.resize((new_w, new_h))
    canvas = Image.new("L", (size, size), 0)
    canvas.paste(small, ((size - new_w) // 2, (size - new_h) // 2))
    return canvas


def collect_stages(path: str, backbone: str = "efficientnet_b0", size: int = 224,
                   resize_mode: str = "pad", preprocess_fn=None, augment_fn=None,
                   channel_spec=None, n_draws: int = 3, verify: bool = True) -> list:
    """
    เดินท่อเตรียมภาพทีละขั้นกับไฟล์ 1 ใบ แล้วคืนภาพของทุกขั้นออกมา

    output: list ของ (ชื่อขั้นตอน, ตารางตัวเลขของภาพ, ข้อความบอกช่วงค่า)
            ขั้นสุดท้าย ("model input") เป็นสิ่งที่โมเดลเห็นจริงหลัง normalize แล้ว
            ซึ่งมีค่าติดลบได้ จึงวาดด้วยการปรับสเกลเพื่อให้ตาคนดูออกเท่านั้น

    verify = ตรวจว่าผลสุดท้ายของเส้นทางที่เดินทีละขั้นในนี้ ตรงกับที่ prepare_image()
             ของจริงคำนวณได้หรือไม่ (เทียบเฉพาะเส้นทางที่ไม่มี augment เพราะ augment
             สุ่มทุกครั้ง เทียบตรงๆ ไม่ได้) — กันภาพที่วาดออกมาไม่ตรงกับของจริง
    """
    stages = []

    # ขั้น 0: ภาพดิบตามที่อยู่ในไฟล์
    raw = Image.open(path).convert("L")
    stages.append(("1. raw crop", np.array(raw), f"{raw.size[0]}x{raw.size[1]} px"))

    # ขั้น 1: preprocess (คงที่ ใช้กับทุก split)
    after_pre = preprocess_fn(raw) if preprocess_fn is not None else raw
    stages.append(("2. preprocess", np.array(after_pre),
                   "no-op" if preprocess_fn is None else "applied"))

    # ขั้น 2: resize / pad ให้เป็นจัตุรัส
    after_resize = _resize_like_transforms(after_pre, size, resize_mode)
    stages.append((f"3. resize ({resize_mode})", np.array(after_resize), f"{size}x{size} px"))

    # ขั้น 3: augment (สุ่ม ทำเฉพาะชุด train) — สุ่มหลายครั้งเพื่อให้เห็นความหลากหลาย
    if augment_fn is not None:
        for i in range(n_draws):
            stages.append((f"4. augment #{i + 1}", np.array(augment_fn(after_resize)), "random"))
    else:
        stages.append(("4. augment", np.array(after_resize), "no-op"))

    # ขั้น 4: normalize = สิ่งที่โมเดลเห็นจริง (ใช้เส้นทางจริงจาก transforms.py)
    final = prepare_image(path, backbone=backbone, size=size,
                          preprocess_fn=preprocess_fn, resize_mode=resize_mode,
                          augment_fn=None, channel_spec=channel_spec)

    # ถ้าใช้สูตรช่องสีแบบผสม ต้องแยกให้เห็นทีละช่อง ไม่งั้นจะไม่รู้เลยว่าแต่ละช่อง
    # มีอะไรอยู่ (ซึ่งเป็นสาระสำคัญทั้งหมดของการเปลี่ยนสูตรช่องสี)
    spec = tuple(channel_spec) if channel_spec is not None else ("gray", "gray", "gray")
    if len(set(spec)) == 1:
        ch0 = final[0]
        stages.append(("5. model input", ch0, f"range {ch0.min():+.2f} to {ch0.max():+.2f}"))
    else:
        for i, cname in enumerate(spec):
            ch = final[i]
            stages.append((f"5. ch{i}: {cname}", ch, f"{ch.min():+.2f} to {ch.max():+.2f}"))

    # --- ตรวจว่าภาพที่วาดตรงกับท่อจริง ---
    # ย้อน normalize ของขั้นสุดท้ายกลับเป็น 0-255 แล้วเทียบกับผลของขั้น resize ในนี้
    # ถ้าตรงกัน แปลว่าที่วาดมาทุกขั้นสะท้อนของจริง ไม่ได้วาดคนละเส้นทาง
    if verify and spec[0] == "gray":
        back = (final[0] * IMAGENET_STD[0] + IMAGENET_MEAN[0]) * 255.0
        diff = np.abs(back - np.array(after_resize, dtype=np.float32)).max()
        if diff > 1.5:      # เผื่อความคลาดเคลื่อนจากการปัดเศษ float
            raise RuntimeError(
                f"ภาพที่วาดไม่ตรงกับท่อจริง (ต่างกันสูงสุด {diff:.2f} ระดับความสว่าง) — "
                f"แปลว่า _resize_like_transforms() ใน pipeline_viz.py ไม่ตรงกับ "
                f"transforms.py แล้ว ต้องแก้ให้ตรงกันก่อนเชื่อภาพที่ได้"
            )

    return stages


# ============================================================================
# ส่วนที่ 2: วาดตารางภาพ
# ============================================================================

def plot_pipeline_stages(samples: list, out_path: Path, title: str = ""):
    """
    วาดตาราง: แถว = ตัวอย่างภาพ, คอลัมน์ = ขั้นตอนในท่อ

    samples = list ของ (ชื่อกำกับแถว, ผลจาก collect_stages())
    """
    n_rows = len(samples)
    n_cols = max(len(s[1]) for s in samples)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.1 * n_cols, 2.5 * n_rows),
                             squeeze=False)

    for r, (row_label, stages) in enumerate(samples):
        for c in range(n_cols):
            ax = axes[r][c]
            ax.set_xticks([]); ax.set_yticks([])

            if c >= len(stages):
                ax.axis("off")
                continue

            name, arr, note = stages[c]
            # ตรึงสเกลทุกขั้นให้เทียบกันได้ตรงๆ
            # ขั้นสุดท้ายผ่าน normalize มาแล้ว ซึ่งเป็นการแปลง "เชิงเส้น" ล้วนๆ
            # ถ้าปล่อยให้ matplotlib ปรับสเกลอัตโนมัติ ภาพจะดูเปลี่ยนไปทั้งที่จริง
            # ไม่ได้เปลี่ยนอะไรเลยนอกจากตัวเลข — จึงคำนวณขอบเขตที่ 0 กับ 255 เดิม
            # แปลงไปเป็นเท่าไหร่ แล้วใช้เป็น vmin/vmax เพื่อให้เห็นตรงกับความจริงว่า
            # "หน้าตาเหมือนขั้นก่อนหน้าเป๊ะ ต่างแค่สเกลตัวเลขที่โมเดลอ่าน"
            if name.startswith("5."):
                lo = (0.0 - IMAGENET_MEAN[0]) / IMAGENET_STD[0]
                hi = (1.0 - IMAGENET_MEAN[0]) / IMAGENET_STD[0]
                ax.imshow(arr, cmap="gray", vmin=lo, vmax=hi)
            else:
                ax.imshow(arr, cmap="gray", vmin=0, vmax=255)

            if r == 0:
                ax.set_title(name, fontsize=9)
            if c == 0:
                ax.set_ylabel(row_label, fontsize=9)
            ax.set_xlabel(note, fontsize=7, labelpad=2)

    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97 if title else 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ============================================================================
# ส่วนที่ 3: ฮิสโทแกรมและตัวเลขสรุปรายขั้นตอน
# ============================================================================

def _bone_values(arr: np.ndarray, is_final: bool) -> np.ndarray:
    """
    ดึงค่าเฉพาะพิกเซลกระดูกออกมา (ตัดพื้นหลังทิ้ง)

    ขั้นสุดท้ายผ่าน normalize มาแล้ว พื้นหลัง 0 จึงกลายเป็นค่าติดลบค่าหนึ่ง
    (ไม่ใช่ 0 อีกต่อไป) ต้องคำนวณย้อนว่าพื้นหลังกลายเป็นเท่าไหร่แล้วตัดตรงนั้น
    """
    if not is_final:
        return arr[arr > 0].astype(np.float32)
    bg = (0.0 - IMAGENET_MEAN[0]) / IMAGENET_STD[0]     # ค่าที่พื้นหลังกลายเป็น
    return arr[arr > bg + 1e-4].astype(np.float32)


def plot_pipeline_histograms(samples: list, out_path: Path, title: str = ""):
    """
    วาดการกระจายค่าความสว่างของ "พิกเซลกระดูก" แยกตามขั้นตอน

    รวมทุกตัวอย่างเข้าด้วยกันในกราฟเดียวต่อขั้น เพื่อให้เห็นภาพรวมว่าแต่ละขั้น
    ดันการกระจายไปทางไหน (เช่น normalize ยืดให้เต็มสเกล, CLAHE ทำให้แบนลง)
    """
    stage_names = [s[0] for s in samples[0][1]]
    n = len(stage_names)
    fig, axes = plt.subplots(1, n, figsize=(2.6 * n, 2.8), squeeze=False)

    for c, name in enumerate(stage_names):
        ax = axes[0][c]
        vals = np.concatenate([_bone_values(st[c][1], name.startswith("5."))
                               for _, st in samples if c < len(st)])
        ax.hist(vals, bins=48, color="#4C78A8")
        ax.set_title(name, fontsize=9)
        ax.set_yticks([])
        ax.tick_params(axis="x", labelsize=7)
        ax.set_xlabel(f"mean {vals.mean():.1f}\nsd {vals.std():.1f}", fontsize=7)

    if title:
        fig.suptitle(title + "  (bone pixels only)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.86 if title else 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def save_pipeline_stats(samples: list, out_path: Path) -> pd.DataFrame:
    """
    เขียนตัวเลขสรุปรายขั้นตอนลง CSV — เอาไปใส่ตารางในเล่มได้ตรงๆ

    คอลัมน์ saturated_pct สำคัญเป็นพิเศษ: บอกว่าขั้นนั้นดันพิกเซลไปชนเพดาน
    จนรายละเอียดหายไปกี่เปอร์เซ็นต์ (ตัวเลขนี้คือสิ่งที่จับปัญหา augment แรงเกินได้)
    """
    rows = []
    for name_idx, name in enumerate([s[0] for s in samples[0][1]]):
        is_final = name.startswith("5.")
        vals = np.concatenate([_bone_values(st[name_idx][1], is_final)
                               for _, st in samples if name_idx < len(st)])
        rows.append({
            "stage": name,
            "n_bone_px": len(vals),
            "mean": round(float(vals.mean()), 3),
            "sd": round(float(vals.std()), 3),
            "p1": round(float(np.percentile(vals, 1)), 3),
            "p99": round(float(np.percentile(vals, 99)), 3),
            "saturated_pct": round(float((vals >= 255).mean() * 100), 3) if not is_final else np.nan,
        })

    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return df


# ============================================================================
# ส่วนที่ 4: จุดเข้าเดียวที่ train.py กับ show_pipeline.py เรียกใช้
# ============================================================================

def pick_samples(df: pd.DataFrame, n_per_grade: int = 1, seed: int = 0) -> list:
    """
    สุ่มเลือกตัวอย่างให้ครบทุก grade — เพื่อให้เห็นว่าท่อทำอะไรกับภาพแต่ละแบบ
    (ปล้องปกติกับปล้องที่ยุบรุนแรงมีความสว่าง/รูปทรงต่างกันมาก ผลของ CLAHE
    หรือ augment จึงไม่เหมือนกัน ต้องดูให้ครบ)
    """
    rng = np.random.RandomState(seed)
    out = []
    for g in sorted(df["grade_4class"].unique()):
        sub = df[df["grade_4class"] == g]
        if len(sub) == 0:
            continue
        for i in rng.choice(len(sub), min(n_per_grade, len(sub)), replace=False):
            row = sub.iloc[i]
            label = f"{GRADE_NAMES.get(int(g), g)}\n{row['level_name']}"
            out.append((label, row["crop_path"]))
    return out


def generate_pipeline_report(df: pd.DataFrame, out_dir: Path, prefix: str = "pipeline",
                             backbone: str = "efficientnet_b0", size: int = 224,
                             resize_mode: str = "pad", preprocess_fn=None, augment_fn=None,
                             channel_spec=None,
                             preprocess_name: str = "none", augment_name: str = "none",
                             channels_name: str = "gray3",
                             n_per_grade: int = 1, n_draws: int = 3, seed: int = 0) -> Path:
    """
    สร้างรายงานท่อเตรียมภาพครบชุด (ภาพ + ฮิสโทแกรม + ตัวเลข) ในคำสั่งเดียว

    เรียกจาก train.py ตอนเริ่มเทรน เพื่อให้ทุกการทดลองมีบันทึกไว้เสมอว่า
    "รอบนี้ป้อนภาพหน้าตาแบบไหนเข้าไป" โดยไม่ต้องจำเอง

    คืน path ของโฟลเดอร์ที่เขียนไฟล์ลงไป
    """
    picks = pick_samples(df, n_per_grade=n_per_grade, seed=seed)
    samples = []
    for label, path in picks:
        stages = collect_stages(path, backbone=backbone, size=size, resize_mode=resize_mode,
                                preprocess_fn=preprocess_fn, augment_fn=augment_fn,
                                channel_spec=channel_spec, n_draws=n_draws)
        samples.append((label, stages))

    title = (f"preprocess={preprocess_name}  augment={augment_name}  "
             f"resize={resize_mode}  channels={channels_name}")
    plot_pipeline_stages(samples, out_dir / f"{prefix}_stages.png", title=title)
    plot_pipeline_histograms(samples, out_dir / f"{prefix}_histograms.png", title=title)
    save_pipeline_stats(samples, out_dir / f"{prefix}_stats.csv")
    return out_dir
