"""
show_transform_catalog.py — แคตตาล็อกภาพ "ทุกตัวเลือก" ของ preprocess และ augment
(รันได้เลย ไม่ต้องเทรน ไม่ต้องมี config)

ต่างจาก show_pipeline.py อย่างไร:
    show_pipeline.py            ดู 1 config -> ภาพเดินผ่านท่อทีละขั้น
    show_transform_catalog.py   ดู "ทุกตัวเลือกที่มี" เทียบกันในภาพเดียว

ใช้ตอบคำถามว่า "ถ้าเลือก preprocess ตัวนี้ / augment ชุดนี้ ภาพที่โมเดลเห็นจะ
หน้าตายังไง" โดยไม่ต้องไปไล่เปิด config ทีละไฟล์

output (ลงที่ outputs/catalog/):
    preprocess_catalog.png       แถว = ตัวอย่างแต่ละ grade, คอลัมน์ = preprocess ทุกตัว
    preprocess_diff.png          เหมือนกันแต่แสดง "ส่วนต่างจากภาพดิบ" ขยาย 4 เท่า
    augment_transforms.png       transform พื้นฐาน 4 ตัว x การสุ่มหลายครั้ง (บังคับ p=1.0)
    augment_presets.png          ชุดสำเร็จ 4 ชุด x การสุ่มหลายครั้ง (ใช้ p จริงตามโค้ด)
    augment_diff.png             ส่วนต่างจากภาพก่อน augment ขยาย 4 เท่า
    catalog_stats.csv            ตัวเลขกำกับทุกช่องในภาพ (mean/sd/p1/p99/อิ่มตัว)

ลำดับที่ยึดตามท่อจริงใน transforms.py เป๊ะ:
    เปิดไฟล์ -> preprocess -> resize/pad -> augment
จึงทำ preprocess บน "crop ดิบ" (ก่อนย่อ) และทำ augment บน "ภาพที่ pad แล้ว"
ถ้าสลับลำดับตรงนี้ ภาพที่ได้จะไม่ใช่สิ่งที่โมเดลเห็นจริง

หมายเหตุเรื่องภาษา: ข้อความในกราฟใช้ภาษาอังกฤษ ตามธรรมเนียมเดียวกับ reporting.py
และ pipeline_viz.py (matplotlib ไม่มีฟอนต์ไทยติดมาให้)

USAGE:
    python scripts/show_transform_catalog.py
    python scripts/show_transform_catalog.py --seed 7 --draws 6 --size 224
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from core.stage2 import augment as A
from core.stage2.dataset import load_split_csv
from core.stage2.preprocessing import PREPROCESS_FNS

GRADE_NAMES = {0: "normal", 1: "mild", 2: "moderate", 3: "severe"}

# preprocess ที่จะเอามาเทียบ — เรียงจาก "ไม่ทำอะไร" ไปหาตัวที่ต่อกันหลายขั้น
# ตัดตัวว่าง (none / no_preprocess) ออก เพราะคอลัมน์แรกคือภาพดิบอยู่แล้ว
PREPROCESS_ORDER = [
    "normalize", "clahe", "normalize_clahe",
    "destripe", "destripe_clahe",
    "unsharp", "flatten",
    "destripe_unsharp", "destripe_flatten",
]

# transform พื้นฐาน — บังคับ p=1.0 เพื่อให้เห็นผลทุกช่อง
# (ในชุดจริง p < 1 บางช่องจึงจะเป็นภาพเดิม ซึ่งดูได้จาก augment_presets.png แทน)
BASE_TRANSFORMS = [
    ("brightness/contrast\n(+-0.20, p=1.0)", lambda im: A.random_brightness_contrast(im, 0.2, 0.2, p=1.0)),
    ("gamma\n(0.85-1.15, p=1.0)", lambda im: A.random_gamma(im, 0.85, 1.15, p=1.0)),
    ("noise\n(sigma 2-8, p=1.0)", lambda im: A.random_noise(im, 2.0, 8.0, p=1.0)),
    ("shift+rotate\n(+-7deg, +-5%, p=1.0)", lambda im: A.random_shift_rotate(im, 7.0, 0.05, p=1.0)),
    # --- 3 ตัวที่เอกสารระบุว่าห้ามใช้ เพราะเปลี่ยนรูปทรงซึ่ง label อ้างอิงโดยตรง ---
    # ต้องดูด้วยตาก่อนเอาไป bake ลงไฟล์จริง ว่าภาพที่ได้ยัง "เป็นกระดูก" อยู่ไหม
    # และเห็นชัดไหมว่ามันเปลี่ยนสิ่งที่ Genant ใช้ตัดสิน
    ("FLIP h+v\n(SHAPE-BREAKING)", lambda im: A.random_flip(im, p_h=1.0, p_v=1.0)),
    ("SCALE\n(0.85-1.15, SHAPE-BREAKING)", lambda im: A.random_scale(im, 0.85, 1.15, p=1.0)),
    ("ELASTIC\n(a=8 s=12, SHAPE-BREAKING)", lambda im: A.random_elastic(im, 8.0, 12.0, p=1.0)),
]

PRESET_ORDER = ["intensity", "geometric", "standard", "strong", "shape", "standard_shape"]


# ============================================================================
# ส่วนที่ 1: เครื่องมือร่วม
# ============================================================================

def resize_pad(img: Image.Image, size: int) -> Image.Image:
    """
    ย่อ+เติมขอบดำแบบเดียวกับ _prepare_standard(resize_mode="pad") ใน transforms.py

    เขียนซ้ำที่นี่เพราะ transforms.py คืนเฉพาะผลสุดท้ายหลัง normalize แล้ว
    ไม่ได้คืนภาพระหว่างทางให้เอามาวาด
    """
    w, h = img.size
    scale = size / max(w, h)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    small = img.resize((new_w, new_h))
    canvas = Image.new("L", (size, size), 0)
    canvas.paste(small, ((size - new_w) // 2, (size - new_h) // 2))
    return canvas


def bone_stats(arr: np.ndarray) -> dict:
    """
    สถิติของ "พิกเซลกระดูก" เท่านั้น (ค่า > 0) ตามกติกาข้อ 1 ของ preprocessing.py

    ถ้าเอาพื้นดำ 23% ของกรอบมาคิดด้วย ค่าเฉลี่ยจะต่ำเกินจริงและ percentile จะเลื่อน
    ทำให้เทียบระหว่าง preprocess ไม่ได้เลย
    """
    bone = arr > 0
    if not bone.any():
        return {"mean": 0.0, "sd": 0.0, "p1": 0.0, "p99": 0.0, "sat_pct": 0.0}
    v = arr[bone].astype(np.float32)
    return {
        "mean": float(v.mean()),
        "sd": float(v.std()),
        "p1": float(np.percentile(v, 1)),
        "p99": float(np.percentile(v, 99)),
        # สัดส่วนพิกเซลที่ชนเพดาน 255 — ตัวชี้วัดที่จับได้ว่า transform แรงเกินจนภาพขาวโพลน
        "sat_pct": float((v >= 254.5).mean() * 100.0),
    }


def pick_samples(df: pd.DataFrame, seed: int) -> list:
    """
    เลือกตัวอย่าง 1 ใบต่อ 1 grade — สุ่มจากใบที่ขนาดอยู่ในช่วงกลางๆ ของชุด

    ที่ไม่สุ่มล้วน เพราะ crop ที่เล็กผิดปกติจะเห็นรายละเอียดไม่ออกเลยหลังย่อ
    ทำให้ภาพแคตตาล็อกดูไม่รู้เรื่องทั้งแถว
    """
    out = []
    for g in [0, 1, 2, 3]:
        sub = df[df["grade_4class"] == g].copy()
        if len(sub) == 0:
            continue
        sizes = sub["crop_path"].apply(lambda p: Image.open(p).size)
        sub["area"] = [w * h for (w, h) in sizes]
        lo, hi = sub["area"].quantile([0.4, 0.6])
        mid = sub[sub["area"].between(lo, hi)]
        mid = mid if len(mid) else sub
        row = mid.sample(1, random_state=seed).iloc[0]
        out.append((g, row["crop_path"]))
    return out


def _row_profile(arr: np.ndarray) -> np.ndarray:
    """ความสว่างเฉลี่ยรายแถว คิดเฉพาะพิกเซลกระดูก (แถวที่กระดูกน้อยกว่า 10 px ตัดทิ้ง)"""
    bone = arr > 0
    counts = bone.sum(axis=1)
    sums = np.where(bone, arr, 0.0).sum(axis=1)
    ok = counts >= 10
    return sums[ok] / counts[ok]


def _jitter(profile: np.ndarray) -> float:
    """
    "การกระตุกรายแถว" = ส่วนที่โปรไฟล์เบี่ยงจาก median filter หน้าต่าง 5 แถว

    กายวิภาคจริงเปลี่ยนความสว่างแบบค่อยเป็นค่อยไปหรือแบบขั้นบันได (endplate)
    ทั้งสองอย่างรอด median filter ส่วนที่เหลือจึงเป็นการกระตุกทีละแถวของเครื่อง
    """
    if len(profile) < 5:
        return 0.0
    pad = np.pad(profile, 2, mode="edge")
    smooth = np.array([np.median(pad[i:i + 5]) for i in range(len(profile))])
    return float(np.std(profile - smooth))


def show(ax, arr: np.ndarray, title: str = "", sub: str = "", diff: bool = False):
    """
    วาด 1 ช่อง

    โหมด diff ปรับสเกลตาม "ค่าสูงสุดของช่องนั้นเอง" ไม่ใช่สเกลร่วม — เพราะความแรง
    ของแต่ละ transform ต่างกันเป็นสิบเท่า (destripe แก้ไม่กี่ระดับ ส่วน normalize
    แก้เป็นร้อยระดับ) ถ้าใช้สเกลร่วม ช่องที่แรงจะขาวโพลนและช่องที่เบาจะดำสนิท
    มองไม่เห็นลวดลายของทั้งคู่ ซึ่งลวดลายคือสิ่งที่ต้องดู ส่วนความแรงอ่านจากตัวเลข
    ใต้ภาพแทน
    """
    if diff:
        ax.imshow(arr, cmap="magma", vmin=0, vmax=max(float(arr.max()), 1e-6))
    else:
        ax.imshow(arr, cmap="gray", vmin=0, vmax=255)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#bbbbbb")
    if title:
        ax.set_title(title, fontsize=8, pad=4)
    if sub:
        ax.set_xlabel(sub, fontsize=6.5, labelpad=2)


# ============================================================================
# ส่วนที่ 2: แคตตาล็อก preprocess
# ============================================================================

def build_preprocess_catalog(samples: list, size: int, out_dir: Path) -> list:
    """
    แถว = ตัวอย่างแต่ละ grade, คอลัมน์ = raw + preprocess ทุกตัวใน PREPROCESS_ORDER

    ทำ preprocess บน crop ดิบก่อนย่อ (ตรงตามท่อจริง) แล้วค่อยย่อเพื่อแสดงผล
    """
    cols = ["raw"] + PREPROCESS_ORDER
    n_rows, n_cols = len(samples), len(cols)
    rows = []

    fig_a, ax_a = plt.subplots(n_rows, n_cols, figsize=(1.75 * n_cols, 2.1 * n_rows), squeeze=False)
    fig_b, ax_b = plt.subplots(n_rows, n_cols, figsize=(1.75 * n_cols, 2.1 * n_rows), squeeze=False)

    for r, (grade, path) in enumerate(samples):
        raw = Image.open(path).convert("L")
        base = np.array(resize_pad(raw, size), dtype=np.float32)

        for c, name in enumerate(cols):
            img = raw if name == "raw" else PREPROCESS_FNS[name](raw)
            arr = np.array(resize_pad(img, size), dtype=np.float32)
            st = bone_stats(arr)

            title = name if r == 0 else ""
            sub = f"mean {st['mean']:.0f} | sd {st['sd']:.1f} | p1-p99 {st['p99'] - st['p1']:.0f}"
            show(ax_a[r][c], arr, title, sub)

            # ภาพส่วนต่าง: |หลัง - ก่อน| — ทำให้ผลที่ตาแทบมองไม่เห็นกลายเป็นมองเห็น
            # ลวดลายที่โผล่มาคือ "สิ่งที่ฟังก์ชันนั้นไปแตะ" ซึ่งบอกได้ชัดกว่าภาพผลลัพธ์เอง
            d = np.abs(arr - base)
            d[arr <= 0] = 0
            show(ax_b[r][c], d, title, f"max diff {d.max():.0f} levels", diff=True)

            if c == 0:
                for ax in (ax_a[r][0], ax_b[r][0]):
                    ax.set_ylabel(f"grade {grade}\n{GRADE_NAMES[grade]}", fontsize=9)

            rows.append({"figure": "preprocess", "sample": GRADE_NAMES[grade],
                         "variant": name, **st,
                         "max_diff_vs_raw": float(np.abs(arr - base).max())})

    fig_a.suptitle("Preprocess catalog — deterministic, applied to train/val/test alike",
                   fontsize=12, y=0.995)
    fig_b.suptitle("What each preprocess touches: |after - raw|, each panel scaled to its own max",
                   fontsize=12, y=0.995)
    for f, p in [(fig_a, "preprocess_catalog.png"), (fig_b, "preprocess_diff.png")]:
        f.tight_layout(rect=(0, 0, 1, 0.98))
        f.savefig(out_dir / p, dpi=130)
        plt.close(f)

    return rows


# ============================================================================
# ส่วนที่ 3: แคตตาล็อก augment
# ============================================================================

def build_augment_catalog(sample_path: str, grade: int, size: int, draws: int,
                          seed: int, out_dir: Path) -> list:
    """
    augment ทำ **หลัง** resize/pad เสมอ (crop ตัดชิดกระดูก ถ้าหมุนก่อน pad มุมจะหาย)
    จึงเตรียมภาพให้เป็น 224x224 ก่อน แล้วค่อยป้อนเข้า transform

    วาด 3 รูป: transform พื้นฐาน (บังคับ p=1) / ชุดสำเร็จ (p จริง) / ส่วนต่างของชุดสำเร็จ
    """
    raw = Image.open(sample_path).convert("L")
    padded = resize_pad(raw, size)
    base = np.array(padded, dtype=np.float32)
    rows = []

    def draw_grid(items, fname, title, with_diff=False):
        n_rows, n_cols = len(items), draws + 1
        fig, ax = plt.subplots(n_rows, n_cols, figsize=(1.75 * n_cols, 2.1 * n_rows), squeeze=False)
        for r, (label, fn) in enumerate(items):
            st0 = bone_stats(base)
            show(ax[r][0], base, "before" if r == 0 else "",
                 f"mean {st0['mean']:.0f} | sd {st0['sd']:.1f}")
            ax[r][0].set_ylabel(label, fontsize=8)
            for c in range(draws):
                # ตั้ง seed ต่อช่อง ให้รันซ้ำได้ผลเดิม และให้ทุกแถวใช้ชุดเลขสุ่มเดียวกัน
                # (แถวที่แรงกว่าจึงต่างกันเพราะพารามิเตอร์ ไม่ใช่เพราะบังเอิญสุ่มได้คนละค่า)
                np.random.seed(seed * 1000 + c)
                arr = np.array(fn(padded), dtype=np.float32)
                st = bone_stats(arr)
                d = np.abs(arr - base)
                unchanged = d.max() < 0.5
                sub = ("unchanged (p roll failed)" if unchanged
                       else f"mean {st['mean']:.0f} | sd {st['sd']:.1f} | sat {st['sat_pct']:.2f}%")
                if with_diff:
                    dd = d.copy()
                    dd[arr <= 0] = 0
                    show(ax[r][c + 1], dd, f"draw {c + 1}" if r == 0 else "",
                         f"max diff {d.max():.0f} levels", diff=True)
                else:
                    show(ax[r][c + 1], arr, f"draw {c + 1}" if r == 0 else "", sub)
                rows.append({"figure": fname.replace(".png", ""),
                             "sample": GRADE_NAMES[grade], "variant": label.replace("\n", " "),
                             "draw": c + 1, **st, "max_diff_vs_before": float(d.max()),
                             "unchanged": bool(unchanged)})
        fig.suptitle(title, fontsize=12, y=0.995)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(out_dir / fname, dpi=130)
        plt.close(fig)

    draw_grid(BASE_TRANSFORMS, "augment_transforms.png",
              f"Base transforms, p forced to 1.0 — sample: grade {grade} ({GRADE_NAMES[grade]})")
    draw_grid(BASE_TRANSFORMS, "augment_transforms_diff.png",
              "What each base transform touches: |after - before|, each panel scaled to its own max",
              with_diff=True)

    presets = [(n, A.AUGMENT_FNS[n]) for n in PRESET_ORDER]
    draw_grid(presets, "augment_presets.png",
              f"Augment presets with real p values — sample: grade {grade} ({GRADE_NAMES[grade]})")
    draw_grid(presets, "augment_diff.png",
              "What each preset touches: |after - before|, each panel scaled to its own max",
              with_diff=True)

    return rows


# ============================================================================
# ส่วนที่ 3b: ภาพพิเศษ — ทำไมต้องมี destripe
# ============================================================================

def build_destripe_evidence(samples: list, size: int, out_dir: Path):
    """
    ลายเส้นแนวนอนแทบมองไม่เห็นในภาพดิบ เพราะคอนทราสต์ของมันต่ำมาก
    การดู "raw vs destripe" ตรงๆ จึงเห็นแทบไม่ต่างเลย (ดูจาก preprocess_catalog.png)

    รูปนี้แก้ปัญหานั้นด้วยการดูทั้งคู่ "ผ่านแว่นขยาย" ตัวเดียวกัน คือ normalize
    ซึ่งยืดคอนทราสต์จนลายเส้นโผล่ออกมาให้ตาเห็น — เทียบ 2 คอลัมน์ขวาแล้วจะเห็นว่า
    destripe ลบอะไรออกไปจริงบ้าง (นี่คือเหตุผลทั้งหมดที่ destripe ถูกเขียนขึ้นมา:
    CLAHE ขยายลายเส้นนี้ +76% ถ้าไม่ลบก่อน สิ่งที่ CLAHE ขยายให้ก็คือ artifact)
    """
    from core.stage2.preprocessing import destripe, normalize

    cols = [("raw", lambda im: im),
            ("normalize(raw)\nstripes revealed", normalize),
            ("normalize(destripe(raw))", lambda im: normalize(destripe(im)))]

    fig, ax = plt.subplots(len(samples), len(cols) + 1,
                           figsize=(2.9 * (len(cols) + 1), 3.0 * len(samples)), squeeze=False)
    for r, (grade, path) in enumerate(samples):
        raw = Image.open(path).convert("L")
        for c, (label, fn) in enumerate(cols):
            arr = np.array(resize_pad(fn(raw), size), dtype=np.float32)
            st = bone_stats(arr)
            show(ax[r][c], arr, label if r == 0 else "", f"sd {st['sd']:.1f}")
        ax[r][0].set_ylabel(f"grade {grade}\n{GRADE_NAMES[grade]}", fontsize=9)

        # คอลัมน์สุดท้าย: โปรไฟล์ความสว่างรายแถว — พื้นที่ที่ destripe ทำงานจริง
        # ตาคนดูภาพแล้วแยกไม่ออกว่าอะไรคือลายเส้น อะไรคือเนื้อกระดูก แต่เส้นนี้แยกให้เห็น
        p_raw = _row_profile(np.array(raw, dtype=np.float32))
        p_ds = _row_profile(np.array(destripe(raw), dtype=np.float32))
        a = ax[r][len(cols)]
        a.plot(p_raw, range(len(p_raw)), lw=0.8, color="#c44", label="raw")
        a.plot(p_ds, range(len(p_ds)), lw=0.8, color="#248", label="destriped")
        a.invert_yaxis(); a.set_xlabel("row mean (bone px)", fontsize=7)
        a.tick_params(labelsize=6)
        if r == 0:
            a.set_title("row-brightness profile", fontsize=8)
            a.legend(fontsize=6, loc="lower right")
        a.text(0.02, 0.02, f"row jitter {_jitter(p_raw):.1f} -> {_jitter(p_ds):.1f}",
               transform=a.transAxes, fontsize=6.5, va="bottom")

    fig.suptitle("Why destripe exists — the artifact lives in the row profile, not in what the eye sees",
                 fontsize=12, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_dir / "destripe_evidence.png", dpi=130)
    plt.close(fig)


# ============================================================================
# ส่วนที่ 4: main
# ============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split_dir", type=Path, default=Path("data/processed/splits"))
    ap.add_argument("--variant", default="xray_masked")
    ap.add_argument("--split", default="train", choices=["train", "val", "test"])
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--draws", type=int, default=5, help="สุ่ม augment กี่ครั้งต่อแถว")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--aug_grade", type=int, default=1,
                    help="ใช้ grade ไหนเป็นตัวอย่างของภาพ augment (ค่าเริ่มต้น 1 = mild)")
    ap.add_argument("--out_dir", type=Path, default=Path("outputs/catalog"))
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = args.split_dir / f"{args.variant}_{args.split}.csv"
    df = load_split_csv(str(csv_path), task="multiclass")
    print(f"ชุดข้อมูล : {csv_path}  ({len(df)} ปล้อง)")

    samples = pick_samples(df, args.seed)
    for g, p in samples:
        print(f"  grade {g} ({GRADE_NAMES[g]}): {Path(p).name}")

    stats = build_preprocess_catalog(samples, args.size, args.out_dir)
    build_destripe_evidence(samples, args.size, args.out_dir)

    aug_sample = dict(samples).get(args.aug_grade, samples[0][1])
    stats += build_augment_catalog(aug_sample, args.aug_grade, args.size,
                                   args.draws, args.seed + 1, args.out_dir)

    pd.DataFrame(stats).to_csv(args.out_dir / "catalog_stats.csv", index=False)

    print(f"\nเซฟไว้ที่ {args.out_dir}")
    for f in sorted(args.out_dir.iterdir()):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
