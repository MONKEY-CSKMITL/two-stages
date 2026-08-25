"""
make_slide_figs.py — สร้างรูปประกอบทั้งหมดสำหรับสไลด์นำเสนอ (notebooks/slide_outline.md)

รันครั้งเดียวได้รูปครบ 14 รูปที่ outputs/slides/
    ./.venv/Scripts/python.exe .\\scripts\\make_slide_figs.py

เลือกทำเฉพาะบางรูป:
    ./.venv/Scripts/python.exe .\\scripts\\make_slide_figs.py --only 1 2 9

ที่มาของตัวเลข — แยกเป็น 2 ทาง เพื่อให้ตรวจย้อนได้ว่าอันไหนอ่านสด อันไหนพิมพ์มือ:
    อ่านสดจากไฟล์ผล : รูป 1, 2, 8, 9  (metrics.json / probe_results.csv / SEEDS_binary.csv)
    พิมพ์จากรายงาน  : รูป 3-7, 10-14  (dl-sp/REPORT*.md, notes.txt, experiment_summary.md)
                      ค่าคงที่ทั้งหมดรวมไว้ที่ส่วน CONSTANTS ด้านล่าง มีที่มากำกับทุกตัว

หลักการวาดที่ใช้ทุกรูป (ตาม notebooks/slide_outline.md):
    - 1 สีเน้นสำหรับ "ตัวที่กำลังพูดถึง" ที่เหลือสีเทา
    - ใส่ error bar ทุกครั้งที่มีหลาย seed
    - เขียนจำนวนปล้องจริงกำกับ % เสมอ
    - ใช้คำว่า "ปล่อยหลุด / เตือนผิด" แทน false negative / false positive
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon
from matplotlib import font_manager

# ---------------------------------------------------------------- paths

ROOT = Path(__file__).resolve().parent.parent          # two-stages/
DLSP = ROOT.parent / "dl-sp"                           # โปรเจกต์ EDA/one-stage
OUT = ROOT / "outputs" / "slides"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- style

# ฟอนต์ไทย: Leelawadee UI มากับ Windows ทุกเครื่อง ถ้าไม่มีค่อยไล่หาตัวสำรอง
_have = {f.name for f in font_manager.fontManager.ttflist}
for _cand in ("Leelawadee UI", "Tahoma", "TH Sarabun New", "Segoe UI"):
    if _cand in _have:
        THAI_FONT = _cand
        break
else:                                                   # pragma: no cover
    THAI_FONT = "DejaVu Sans"
    print("! ไม่พบฟอนต์ไทย ข้อความภาษาไทยจะขึ้นเป็นสี่เหลี่ยม", file=sys.stderr)

plt.rcParams.update({
    # ใส่เป็นลิสต์เพื่อให้ matplotlib ถอยไปหาตัวสำรองเมื่อฟอนต์ไทยไม่มี glyph
    # (เช่น เลขยกกำลังติดลบ กับลูกศร ซึ่ง Leelawadee UI ไม่มี)
    "font.family": [THAI_FONT, "DejaVu Sans"],
    "font.size": 11,
    "axes.edgecolor": "#c9d2d6",
    "axes.linewidth": 0.9,
    "axes.labelcolor": "#3d4d56",
    "text.color": "#111a1f",
    "xtick.color": "#3d4d56",
    "ytick.color": "#3d4d56",
    "xtick.major.size": 0,
    "ytick.major.size": 3,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.bbox": "tight",
    "savefig.dpi": 200,
})

ACCENT = "#0d6b74"      # ตัวที่กำลังพูดถึง
ACCENT_L = "#7fc4c9"
WARN = "#a1521a"        # จุดที่ต้องระวัง
ALERT = "#98342b"       # ข้อสรุปที่ผิด / ตัวเลขที่น่าตกใจ
GREY = "#b6c0c5"        # ทุกอย่างที่เหลือ
GREY_D = "#7d8c93"
INK = "#111a1f"


def _finish(fig, name, note=None):
    """เขียนที่มาของข้อมูลไว้มุมล่างซ้ายทุกรูป แล้วเซฟ"""
    if note:
        fig.text(0.005, 0.005, note, fontsize=7, color="#93a2a8", ha="left", va="bottom")
    path = OUT / name
    fig.savefig(path)
    plt.close(fig)
    print(f"  -> {path.relative_to(ROOT)}")


def _barlabels(ax, bars, labels, fmt_color=INK, dy=0.012, fs=10, weight="bold"):
    for b, t in zip(bars, labels):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + dy, t,
                ha="center", va="bottom", fontsize=fs, fontweight=weight, color=fmt_color)


# ================================================================ CONSTANTS
# ตัวเลขที่พิมพ์จากรายงาน — แก้ที่นี่ที่เดียวถ้ารายงานอัปเดต

# dl-sp/REPORT.md 1.2 — จำนวนปล้องที่แพทย์ระบุว่าหัก ต่อคนไข้ 1 คน
FRACTURES_PER_PATIENT = {"0": 1900, "1": 534, "2": 264, "3": 113, "4": 50, "5": 18, "6+": 21}

# dl-sp/REPORT_distribution.md — อัตราหักรายปล้อง (%)
LEVEL_FX_PCT = {
    "T3": 0.52, "T4": 0.15, "T5": 0.89, "T6": 2.19, "T7": 3.39, "T8": 3.75,
    "T9": 2.66, "T10": 3.74, "T11": 16.44, "T12": 26.23, "L1": 28.61,
    "L2": 11.06, "L3": 8.89, "L4": 5.66, "L5": 3.98,
}
LEVEL_FX_N = {
    "T3": 2, "T4": 1, "T5": 7, "T6": 18, "T7": 28, "T8": 31, "T9": 22, "T10": 31,
    "T11": 136, "T12": 218, "L1": 238, "L2": 92, "L3": 74, "L4": 47, "L5": 33,
}

# dl-sp/REPORT_distribution.md — การกระจาย grade ทั้งชุด
GRADE_COUNTS = {"0 ปกติ": 10790, "1 เล็กน้อย": 317, "2 ปานกลาง": 391, "3 รุนแรง": 269}

# dl-sp/REPORT_distribution.md — selection bias
SELBIAS = {"มี mask\n(833 ภาพ)": 71.1, "ไม่มี mask\n(2,067 ภาพ)": 19.7}

# dl-sp/notes.txt [0] — offset test ยืนยัน mapping ของ mask
OFFSET_TEST = {"-2": 0.819, "-1": 0.896, "0": 0.986, "+1": 0.921, "+2": 0.851}

# two-stages/experiment_summary.md 12.1 — R0 (ไม่ทำอะไร) vs R5 (+destripe)
METRIC_BLINDSPOT = {
    "R0 ตั้งต้น": {"macro_f1": 0.567, "mild_r": 0.217, "mild_n": 10},
    "R5 +destripe": {"macro_f1": 0.567, "mild_r": 0.630, "mild_n": 29},
}
MILD_TEST_N = 46

# dl-sp/notes.txt [7] + REPORT.md 3.2 — เทียบ one-stage กับ crop รายปล้อง
STAGE_COMPARE = [
    ("ภาพทั้งใบ\nEffNet-B0 + 15 heads", 0.3030, None, GREY),
    ("ภาพทั้งใบ\n+ Transformer 15 query", 0.2899, None, GREY),
    ("crop รายปล้อง\nConvNeXt-Tiny (5-fold)", 0.6066, 0.0114, ACCENT),
    ("crop รายปล้อง\nensemble 5 folds", 0.6325, None, ACCENT),
]

# dl-sp/PIPELINE_masks_crop.md — ผลของการขยายข้อมูลด้วย auto-seg
DATASET_GROWTH = {
    "จาก mask มือ\n833 ภาพ": {"total": 11767, "g0": 10790},
    "รวมหลัง auto-seg\n2,900 ภาพ": {"total": 26365, "g0": round(26365 * 0.959)},
}
# รอบยืนยันเคสหักรอบแรก 56 คนไข้
FRACTURE_PASS1 = {"g0": 684, "g1": 23, "g2": 37, "g3": 35}

# dl-sp/notes.txt [4] Q3 — AUC ของ 3 อินพุตที่เทียบกันในรูปที่ 1
SHAPE_VS_TEXTURE_AUC = {"xray_bbox": 0.857, "xray_masked": 0.919, "mask_shape": 0.942}


# ================================================================ FIGURES

def fig01_shape_vs_texture():
    """สไลด์ 22 — X-ray bbox / X-ray masked / เงาขาวดำ ของปล้องเดียวกัน พร้อม AUC

    ประเด็น: ภาพเงาที่ไม่มีเนื้อ X-ray เลย ได้ AUC สูงสุด
    วางเทียบ 2 แถว (grade 0 กับ grade 3) เพื่อให้เห็นว่าสิ่งที่ต่างกันคือเส้นขอบ
    """
    from PIL import Image

    KEYS = ("xray_bbox", "xray_masked", "mask_shape")
    idx = pd.read_csv(DLSP / "emb_index.csv")

    def pick(grade):
        """เลือกปล้องที่ 'ยุบระดับกลางๆ ของ grade นั้น' ไม่ใช่ตัวสุดขั้ว จะได้ยังดูออกว่าเป็นกระดูก"""
        c = idx[(idx.grade == grade) & (idx.level.isin(["T12", "L1"]))].copy()
        c = c.dropna(subset=["aspect_hw"])
        target = c.aspect_hw.median()
        c["d"] = (c.aspect_hw - target).abs()
        for _, r in c.sort_values("d").iterrows():
            pid, lab = f"{int(r.image):04d}", f"L{int(r.label):02d}"
            paths = {k: DLSP / "crops" / pid / f"{pid}_{lab}_{k}.png" for k in KEYS}
            if all(p.exists() for p in paths.values()):
                return pid, r.level, float(r.aspect_hw), paths
        return None

    rows = [(0, pick(0)), (3, pick(3))]
    if any(r[1] is None for r in rows):
        print("  ! หาปล้องตัวอย่างที่มีครบทั้ง 3 แบบไม่เจอ ข้ามรูปนี้")
        return

    col_titles = ["X-ray crop แบบกรอบ\n(bbox)",
                  "X-ray crop ลบพื้นหลัง\n(masked)",
                  "เงา mask ล้วน\nไม่มีเนื้อภาพเลย"]

    fig, axes = plt.subplots(2, 3, figsize=(9.8, 5.2),
                             gridspec_kw={"hspace": 0.30, "wspace": 0.10})
    for row, (grade, (pid, level, ar, paths)) in enumerate(rows):
        for col, key in enumerate(KEYS):
            ax = axes[row][col]
            ax.imshow(np.array(Image.open(paths[key]).convert("L")),
                      cmap="gray", vmin=0, vmax=255, aspect="equal")
            ax.set_xticks([]); ax.set_yticks([])
            best = key == "mask_shape"
            for sp in ax.spines.values():
                sp.set_edgecolor(ACCENT if best else "#c9d2d6")
                sp.set_linewidth(2.4 if best else 1.0)
            if row == 0:
                ax.set_title(col_titles[col], fontsize=11, color=INK, pad=8,
                             fontweight="bold" if best else "normal")
            if col == 0:
                ax.set_ylabel(f"Genant grade {grade}\n{level} · ผู้ป่วย {pid}",
                              fontsize=11, labelpad=12,
                              color=ALERT if grade == 3 else GREY_D,
                              fontweight="bold")

    # แถบ AUC วางใต้รูปทั้งคอลัมน์
    for col, key in enumerate(KEYS):
        best = key == "mask_shape"
        bb = axes[1][col].get_position()
        fig.text(bb.x0 + bb.width / 2, bb.y0 - 0.085, f"AUC {SHAPE_VS_TEXTURE_AUC[key]:.3f}",
                 ha="center", va="top", fontsize=16, fontweight="bold",
                 color=ACCENT if best else GREY_D)

    fig.suptitle("สัญญาณคือ “รูปทรง” ไม่ใช่ “เนื้อภาพ”", fontsize=14.5,
                 fontweight="bold", y=1.01)
    fig.text(0.5, -0.10,
             "คอลัมน์ขวาสุดถูกลบเนื้อ X-ray ออกจนหมด เหลือแต่เส้นขอบ — แต่ได้คะแนนสูงที่สุด",
             ha="center", fontsize=11.5, color=WARN, fontweight="bold")
    _finish(fig, "fig01_shape_vs_texture.png",
            "AUC จาก dl-sp/probe_results.csv (linear probe, StratifiedGroupKFold-5 ระดับคนไข้)")


def fig02_binary_tradeoff():
    """สไลด์ 44 — scatter: ปล่อยหลุด vs เตือนผิด ของ 7 วิธีทำสมดุล

    ประเด็น: อันดับตาม macro F1 กลับหัวกับอันดับตามจำนวนผู้ป่วยที่ปล่อยหลุด
    """
    seeds = pd.read_csv(ROOT / "outputs" / "comparison" / "SEEDS_binary.csv")
    rows = []
    for _, r in seeds.iterrows():
        mp = ROOT / "outputs" / "runs" / r["run"] / "metrics.json"
        cm = json.load(open(mp, encoding="utf-8"))["confusion_matrix"]
        # binary confusion matrix = [[TN, FP], [FN, TP]]
        rows.append({"cond": r["cond"], "macro_f1": r["macro_f1"],
                     "fn": cm[1][0], "fp": cm[0][1]})
    df = pd.DataFrame(rows).groupby("cond").mean(numeric_only=True).reset_index()

    best_recall = df.loc[df.fn.idxmin(), "cond"]
    best_f1 = df.loc[df.macro_f1.idxmax(), "cond"]

    fig, ax = plt.subplots(figsize=(9, 6.4))
    for _, r in df.iterrows():
        hi = r["cond"] in (best_recall, best_f1)
        col = ACCENT if r["cond"] == best_recall else (ALERT if r["cond"] == best_f1 else GREY_D)
        ax.scatter(r.fn, r.fp, s=380 if hi else 210, color=col,
                   edgecolor="white", linewidth=1.6, zorder=3, alpha=0.95 if hi else 0.75)
        ax.annotate(f"{r['cond']}\nF1 {r.macro_f1:.3f}", (r.fn, r.fp),
                    textcoords="offset points", xytext=(0, 20 if hi else 17),
                    ha="center", fontsize=10.5 if hi else 9.5,
                    fontweight="bold" if hi else "normal",
                    color=col if hi else "#5c6c74")

    ax.set_xlabel("ปล่อยผู้ป่วยหลุด  (ปล้องที่หักจริงแต่ทายว่าปกติ · จาก 132 ปล้อง)",
                  fontsize=12, labelpad=10)
    ax.set_ylabel("เตือนผิด  (ปล้องปกติแต่ทายว่าหัก · จาก 1,548 ปล้อง)",
                  fontsize=12, labelpad=10)
    ax.set_title("macro F1 จัดอันดับกลับหัวกับสิ่งที่งานคัดกรองต้องการ",
                 fontsize=14.5, fontweight="bold", pad=16)

    ax.annotate("", xy=(0.03, 0.06), xytext=(0.30, 0.06), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color=ACCENT, lw=2.2))
    ax.text(0.315, 0.06, "ดีขึ้นสำหรับการคัดกรอง", transform=ax.transAxes,
            fontsize=11, color=ACCENT, fontweight="bold", va="center")

    ax.text(0.98, 0.97,
            f"เขียว = ปล่อยหลุดน้อยที่สุด ({best_recall})\n"
            f"แดง  = macro F1 สูงที่สุด ({best_f1})",
            transform=ax.transAxes, ha="right", va="top", fontsize=10.5,
            bbox=dict(boxstyle="round,pad=0.5", fc="#f4f7f8", ec="#d5dcdf"))

    ax.grid(alpha=0.25, linestyle=":", zorder=0)
    ax.set_axisbelow(True)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + (hi - lo) * 0.14)   # เผื่อที่ให้ป้ายจุดบนสุดไม่ถูกตัด
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    _finish(fig, "fig02_binary_tradeoff.png",
            "อ่านสดจาก outputs/runs/*/metrics.json (confusion_matrix) · เฉลี่ย 2 seed ต่อเงื่อนไข")


def fig03_metric_blindspot():
    """สไลด์ 40 — macro F1 เท่ากันเป๊ะ แต่ mild recall ต่างกันเกือบ 3 เท่า"""
    names = list(METRIC_BLINDSPOT)
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 5.2))

    ax = axes[0]
    vals = [METRIC_BLINDSPOT[n]["macro_f1"] for n in names]
    bars = ax.bar(names, vals, color=[GREY, GREY], width=0.55)
    _barlabels(ax, bars, [f"{v:.3f}" for v in vals], GREY_D, dy=0.012)
    ax.set_ylim(0, 0.78)
    ax.set_title("macro F1", fontsize=13, fontweight="bold", pad=10)
    ax.text(0.5, 0.86, "เท่ากันเป๊ะ", transform=ax.transAxes, ha="center",
            fontsize=13, color=GREY_D, fontweight="bold")

    ax = axes[1]
    vals = [METRIC_BLINDSPOT[n]["mild_r"] for n in names]
    ns = [METRIC_BLINDSPOT[n]["mild_n"] for n in names]
    bars = ax.bar(names, vals, color=[GREY, ACCENT], width=0.55)
    _barlabels(ax, bars, [f"{v:.3f}\n{n}/{MILD_TEST_N} ปล้อง" for v, n in zip(vals, ns)],
               INK, dy=0.012, fs=10.5)
    ax.set_ylim(0, 0.78)
    ax.set_title("recall ของ grade 1 (mild)", fontsize=13, fontweight="bold", pad=10)
    ax.text(0.5, 0.86, "ต่างกัน 2.9 เท่า", transform=ax.transAxes, ha="center",
            fontsize=13, color=ACCENT, fontweight="bold")

    for ax in axes:
        ax.set_yticks([])
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.tick_params(axis="x", labelsize=11.5)

    fig.suptitle("ตัวชี้วัดรวมมองไม่เห็นสิ่งที่เปลี่ยนไปทั้งหมด",
                 fontsize=14.5, fontweight="bold", y=1.0)
    _finish(fig, "fig03_metric_blindspot.png",
            "two-stages/notebooks/experiment_summary.md หัวข้อ 12.1 (R0 vs R5)")


def fig04_selection_bias():
    """สไลด์ 8 — 833 ภาพที่มี mask ไม่ใช่ตัวอย่างสุ่ม"""
    labels = list(SELBIAS)
    vals = [SELBIAS[k] for k in labels]
    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    bars = ax.bar(labels, vals, color=[ALERT, GREY], width=0.5)
    _barlabels(ax, bars, [f"{v}%" for v in vals], INK, dy=1.2, fs=17)

    ax.set_ylim(0, 92)
    ax.set_ylabel("ผู้ป่วยที่มีกระดูกหัก ≥ 1 ปล้อง", fontsize=12, labelpad=10)
    ax.set_title("ภาพที่มี mask ถูกเลือกจากเคสที่หักเป็นหลัก", fontsize=14.5,
                 fontweight="bold", pad=16)
    ax.tick_params(axis="x", labelsize=12)
    ax.set_yticks([])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)

    ax.annotate("", xy=(0, 76), xytext=(1, 76),
                arrowprops=dict(arrowstyle="<|-|>", color=WARN, lw=1.8))
    ax.text(0.5, 78.5, "ต่างกัน 3.6 เท่า   ·   Mann–Whitney p = 9 × 10⁻¹³³",
            ha="center", fontsize=11.5, color=WARN, fontweight="bold")
    ax.text(0.5, 88, "อายุเฉลี่ยไม่ต่างกันเลย (71.6 vs 71.1 · p = 0.23)",
            ha="center", fontsize=10.5, color=GREY_D)

    fig.text(0.5, -0.045,
             "ห้ามรายงานอัตราการหักจากชุดนี้เป็นค่าความชุก — สูงกว่าประชากรจริงราว 3 เท่า\n"
             "แต่รูปแบบเชิงตำแหน่ง (T11/T12/L1) ยังเชื่อถือได้ เพราะกลุ่มไม่มี mask ก็กระจุกที่เดียวกัน",
             ha="center", fontsize=11, color=ALERT, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.6", fc="#f9e9e6", ec=ALERT, lw=1.1))
    _finish(fig, "fig04_selection_bias.png", "dl-sp/REPORT_distribution.md")


def fig05_genant_morphometry():
    """สไลด์ 26 — วัดความสูง 6 จุดตามนิยาม Genant + ทำไมต้องหมุนเข้ากรอบกระดูกสันหลัง"""
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 4.9))

    def vertebra(ax, ha, hm, hp, title, color, sub=None):
        """วาดปล้องด้านข้าง: ความสูงหน้า/กลาง/หลัง เป็นสัดส่วนของความสูงปกติ"""
        w = 1.0
        xs = [0, w / 2, w]
        tops = [ha, hm, hp]
        pts = [(0, 0), (w, 0), (w, hp), (w / 2, hm), (0, ha)]
        ax.add_patch(Polygon(pts, closed=True, fc=color, ec=INK, lw=1.6, alpha=0.30))
        for x, h, lab in zip(xs, tops, ["Ha", "Hm", "Hp"]):
            ax.plot([x, x], [0, h], color=INK, lw=1.8, zorder=3)
            ax.plot([x - 0.045, x + 0.045], [h, h], color=INK, lw=1.8, zorder=3)
            ax.plot([x - 0.045, x + 0.045], [0, 0], color=INK, lw=1.8, zorder=3)
            ax.text(x, -0.075, lab, ha="center", va="top", fontsize=12, fontweight="bold")
            ax.scatter([x, x], [0, h], s=26, color=ALERT, zorder=4)
        ax.set_xlim(-0.3, 1.3); ax.set_ylim(-0.28, 1.32)
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(title, fontsize=12.5, fontweight="bold", pad=8)
        if sub:
            ax.text(0.5, -0.19, sub, transform=ax.transAxes, ha="center",
                    fontsize=10.5, color=GREY_D)

    vertebra(axes[0], 1.0, 1.0, 1.0, "ปกติ — grade 0", ACCENT_L,
             "Ha ≈ Hm ≈ Hp   ·   จุดวัดจริง 6 จุด")
    vertebra(axes[1], 0.55, 0.78, 1.0, "ยุบรูปลิ่ม — grade 3", "#e0a894",
             "Ha ลดลง > 40%  เทียบกับ Hp")

    ax = axes[2]
    ax.set_xlim(-0.35, 1.45); ax.set_ylim(-0.28, 1.32)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("ทำไมต้องหมุนก่อนวัด", fontsize=12.5, fontweight="bold", pad=8)
    ang = np.deg2rad(-19)
    R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    base = np.array([(0, 0), (1, 0), (1, 0.95), (0.5, 0.78), (0, 0.6)])
    c = base.mean(axis=0)
    rot = (base - c) @ R.T + c + np.array([0.12, 0.05])
    ax.add_patch(Polygon(rot, closed=True, fc="#e0a894", ec=INK, lw=1.6, alpha=0.30))
    ax.plot([0.62, 0.62], [-0.16, 1.12], color=GREY, lw=1.6, ls=":")
    # ป้ายแกนภาพวางไว้ใต้เส้น ส่วนป้ายแกนกระดูกสันหลังอยู่มุมซ้ายบน — กันไม่ให้ชนกัน
    ax.text(0.62, -0.20, "แกน y ของภาพ", ha="center", va="top",
            fontsize=9.5, color=GREY_D)
    d = R @ np.array([0, 1])
    ax.annotate("", xy=(0.62 + d[0] * 0.62, 0.55 + d[1] * 0.62),
                xytext=(0.62 - d[0] * 0.62, 0.55 - d[1] * 0.62),
                arrowprops=dict(arrowstyle="<|-|>", color=ACCENT, lw=2.2))
    # วางป้ายไว้มุมซ้ายบนแล้วลากเส้นประไปหาลูกศร กันไม่ให้ข้อความทับหัวลูกศร
    ax.text(-0.30, 1.30, "แกนของกระดูกสันหลัง\n(ทิศจากปล้องบน-ล่าง)", fontsize=10,
            color=ACCENT, fontweight="bold", ha="left", va="top")
    ax.annotate("", xy=(0.34, 0.94), xytext=(0.10, 1.06),
                arrowprops=dict(arrowstyle="-", color=ACCENT, lw=1.0, ls=":"))

    fig.text(0.5, -0.10,
             "กระดูกสันหลังโค้ง ปล้องแต่ละระดับจึงเอียงไม่เท่ากัน — ความสูงที่วัดตามแกน y ของภาพ ไม่ใช่ความสูงของปล้อง\n"
             "AUC ฟีเจอร์เดี่ยว:   h/w ของกรอบ 0.712   →   วัดในกรอบภาพ 0.807   →   หมุนเข้ากรอบกระดูกสันหลัง 0.871",
             ha="center", fontsize=11.5, color=INK,
             bbox=dict(boxstyle="round,pad=0.6", fc="#e0f0f0", ec=ACCENT, lw=1.1))
    fig.text(0.5, -0.205,
             "ใช้ทิศจากปล้องข้างเคียง ไม่ใช่แกนของปล้องเอง — แกนของตัวเองได้ AUC 0.286 ซึ่งต่ำกว่าการเดาสุ่ม\n"
             "เพราะปล้องที่ยุบเป็นลิ่มจะเอียงแกนของตัวเอง การหมุนตามแกนนั้นจึงหักล้างความผิดรูปที่กำลังจะวัดทิ้ง",
             ha="center", fontsize=10.5, color=WARN)
    _finish(fig, "fig05_genant_morphometry.png",
            "dl-sp/genant_morphometry_v2.py · extract_variant_embeddings.py (docstring)")


def fig06_onestage_vs_crop():
    """สไลด์ 29 — เปลี่ยนหน่วยจากภาพทั้งใบเป็นรายปล้อง แล้วดีขึ้นเท่าตัว"""
    labels = [s[0] for s in STAGE_COMPARE]
    vals = [s[1] for s in STAGE_COMPARE]
    errs = [s[2] if s[2] else 0 for s in STAGE_COMPARE]
    cols = [s[3] for s in STAGE_COMPARE]

    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    bars = ax.bar(labels, vals, color=cols, width=0.6,
                  yerr=errs, capsize=6, error_kw=dict(ecolor=INK, lw=1.4))
    for b, v, e in zip(bars, vals, errs):
        t = f"{v:.3f}" + (f" ± {e:.3f}" if e else "")
        ax.text(b.get_x() + b.get_width() / 2, v + e + 0.018, t,
                ha="center", fontsize=12, fontweight="bold", color=INK)

    ax.set_ylim(0, 0.78)
    ax.set_ylabel("Macro-F1 บนชุด test", fontsize=12, labelpad=10)
    ax.set_title("รอยหักเป็นสัญญาณเฉพาะที่ — พอ crop ทีละปล้อง คะแนนขึ้นเท่าตัว",
                 fontsize=14, fontweight="bold", pad=16)
    ax.tick_params(axis="x", labelsize=10.5)
    ax.set_yticks([])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)

    ax.axvspan(-0.5, 1.5, color=GREY, alpha=0.10, zorder=0)
    ax.axvspan(1.5, 3.5, color=ACCENT, alpha=0.07, zorder=0)
    ax.text(0.5, 0.72, "one-stage", ha="center", fontsize=11.5, color=GREY_D, fontweight="bold")
    ax.text(2.5, 0.72, "two-stage", ha="center", fontsize=11.5, color=ACCENT, fontweight="bold")

    fig.text(0.5, -0.075,
             "เทียบกันตรงๆ ไม่ได้ 100% — one-stage ทาย 15 ปล้องพร้อมกันจากภาพทั้งใบ\n"
             "ส่วน two-stage ทายทีละปล้อง โดยมี mask ของมนุษย์บอกตำแหน่งให้แล้ว",
             ha="center", fontsize=10.5, color=WARN)
    _finish(fig, "fig06_onestage_vs_crop.png", "dl-sp/notes.txt [7] · dl-sp/REPORT.md 3.2")


def fig07_fractures_per_patient():
    """สไลด์ 4 — 80% ของคนที่หัก หักแค่ 1-2 ปล้อง"""
    labels = list(FRACTURES_PER_PATIENT)
    vals = [FRACTURES_PER_PATIENT[k] for k in labels]
    cols = [GREY] + [ACCENT, ACCENT] + [GREY] * 4

    fig, ax = plt.subplots(figsize=(9, 5.4))
    bars = ax.bar(labels, vals, color=cols, width=0.66)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 28, f"{v:,}",
                ha="center", fontsize=10.5, fontweight="bold", color=INK)

    ax.set_xlabel("จำนวนปล้องที่แพทย์ระบุว่าหัก (ต่อผู้ป่วย 1 คน)", fontsize=12, labelpad=10)
    ax.set_ylabel("จำนวนผู้ป่วย", fontsize=12, labelpad=10)
    ax.set_title("ในกลุ่มที่หัก 1,000 คน มี 798 คน (80%) ที่หักแค่ 1–2 ปล้อง",
                 fontsize=14, fontweight="bold", pad=16)
    ax.set_ylim(0, 2180)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", alpha=0.22, linestyle=":")
    ax.set_axisbelow(True)

    ax.annotate("", xy=(0.72, 640), xytext=(2.28, 640),
                arrowprops=dict(arrowstyle="<|-|>", color=ACCENT, lw=2))
    ax.text(1.5, 700, "798 คน = 80% ของคนที่หัก", ha="center", fontsize=12,
            color=ACCENT, fontweight="bold")

    fig.text(0.5, -0.055,
             "รอยหักจึงเป็นสัญญาณเฉพาะที่ — กินพื้นที่เล็กมากของภาพทั้งใบ",
             ha="center", fontsize=11, color=WARN, fontweight="bold")
    _finish(fig, "fig07_fractures_per_patient.png", "dl-sp/REPORT.md 1.2")


def fig08_linear_probe():
    """สไลด์ 20-21 — ตาราง linear probe เป็นกราฟแท่ง + forest plot ของ 95% CI"""
    probe = pd.read_csv(DLSP / "probe_results.csv")
    order = [
        ("aspect h/w (1 feat)", "h/w อย่างเดียว (1 ตัว)", GREY),
        ("aspect+size+level (5)", "geometry 5 ตัว", GREY),
        ("meddinov3/xray_bbox", "MedDINOv3 / bbox", GREY_D),
        ("dinov3/xray_bbox", "DINOv3 / bbox", GREY_D),
        ("dinov3/xray_masked", "DINOv3 / masked", ACCENT_L),
        ("meddinov3/xray_masked", "MedDINOv3 / masked", ACCENT_L),
        ("dinov3/mask_shape", "DINOv3 / เงา mask", ACCENT_L),
        ("meddinov3/mask_shape", "MedDINOv3 / เงา mask", ACCENT),
    ]
    look = dict(zip(probe.features, probe.auc_fx))

    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.8),
                             gridspec_kw={"width_ratios": [1.15, 1], "wspace": 0.62})

    ax = axes[0]
    names = [o[1] for o in order]
    vals = [look[o[0]] for o in order]
    cols = [o[2] for o in order]
    bars = ax.barh(names, vals, color=cols, height=0.66)
    for b, v in zip(bars, vals):
        ax.text(v + 0.004, b.get_y() + b.get_height() / 2, f"{v:.3f}",
                va="center", fontsize=10.5, fontweight="bold", color=INK)
    ax.set_xlim(0.65, 0.985)
    ax.set_xlabel("AUC (แยก grade ≥ 1 ออกจาก grade 0)", fontsize=11.5, labelpad=8)
    ax.set_title("ป้อนอะไรให้ backbone ที่แช่แข็งไว้", fontsize=13, fontweight="bold", pad=12)
    ax.tick_params(axis="y", labelsize=11)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", alpha=0.22, linestyle=":")
    ax.set_axisbelow(True)

    # forest plot — ตัวเลขจาก dl-sp/compare_report.txt (bootstrap 2,000 รอบ ระดับคนไข้)
    contrasts = [
        ("MedDINOv3 − DINOv3 (masked)", 0.005, -0.003, 0.013, False),
        ("MedDINOv3 − DINOv3 (bbox)", -0.001, -0.011, 0.008, False),
        ("masked − bbox (DINOv3)", 0.055, 0.044, 0.067, True),
        ("masked − bbox (MedDINOv3)", 0.061, 0.050, 0.073, True),
        ("X-ray จริง − เงา (DINOv3)", -0.003, -0.011, 0.004, False),
        ("X-ray จริง − เงา (MedDINOv3)", -0.023, -0.032, -0.015, True),
        ("เงา − geometry baseline", 0.075, 0.061, 0.089, True),
        ("preproc − ภาพดิบ (MedDINO, masked)", -0.012, -0.019, -0.006, True),
        ("16-bit DICOM − เงา (MedDINO)", -0.029, -0.038, -0.020, True),
    ]
    ax = axes[1]
    ys = np.arange(len(contrasts))[::-1]
    for y, (lab, d, lo, hi, sig) in zip(ys, contrasts):
        col = ACCENT if sig and d > 0 else (ALERT if sig and d < 0 else GREY_D)
        ax.plot([lo, hi], [y, y], color=col, lw=2.4, solid_capstyle="round",
                alpha=1.0 if sig else 0.55)
        ax.scatter([d], [y], s=90 if sig else 55, color=col, zorder=3,
                   edgecolor="white", linewidth=1.2, alpha=1.0 if sig else 0.55)
    ax.axvline(0, color=INK, lw=1.2, ls="--", alpha=0.65)
    ax.set_yticks(ys)
    ax.set_yticklabels([c[0] for c in contrasts], fontsize=10.5)
    ax.set_xlabel("ส่วนต่างของ AUC พร้อมช่วงเชื่อมั่น 95%", fontsize=11.5, labelpad=8)
    ax.set_title("เทียบเป็นคู่ (bootstrap 2,000 รอบ ระดับคนไข้)",
                 fontsize=13, fontweight="bold", pad=12)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", alpha=0.22, linestyle=":")
    ax.set_axisbelow(True)
    ax.text(0.99, 0.02, "จาง = ช่วงเชื่อมั่นคร่อม 0 (แยกกันไม่ออก)",
            transform=ax.transAxes, ha="right", fontsize=9.5, color=GREY_D)

    _finish(fig, "fig08_linear_probe.png",
            "dl-sp/probe_results.csv · compare_report.txt · compare_dcm_report.txt")


def fig09_mild_ceiling():
    """สไลด์ 43 — mild recall ของทุกรัน multiclass กระจุกอยู่ที่ 0.30"""
    vals, best = [], (None, -1, None)
    for p in sorted((ROOT / "outputs" / "runs").glob("*/metrics.json")):
        d = json.load(open(p, encoding="utf-8"))
        if d.get("task") != "multiclass":
            continue
        r = d.get("per_grade_recall", {}).get("mild", {}).get("recall")
        if r is None:
            continue
        vals.append(r)
        if r > best[1]:
            best = (d["experiment"], r, d.get("best_epoch"))
    vals = np.array(vals)
    med = float(np.median(vals))
    n_hi = int((vals >= 0.5).sum())

    fig, ax = plt.subplots(figsize=(9.4, 5.6))
    bins = np.arange(0, 0.72, 0.04)
    ax.hist(vals[vals < 0.5], bins=bins, color=GREY, edgecolor="white", linewidth=1.1)
    ax.hist(vals[vals >= 0.5], bins=bins, color=ACCENT, edgecolor="white", linewidth=1.1)

    ax.axvline(med, color=ALERT, lw=2.2, ls="-")
    ax.text(med, ax.get_ylim()[1] * 0.97, f" มัธยฐาน {med:.3f}", color=ALERT,
            fontsize=12, fontweight="bold", va="top")
    ax.axvline(0.5, color=ACCENT, lw=1.8, ls="--")
    ax.text(0.5, ax.get_ylim()[1] * 0.62,
            f" ≥ 0.50 มีแค่ {n_hi}/{len(vals)} รัน ({n_hi/len(vals)*100:.0f}%)",
            color=ACCENT, fontsize=11.5, fontweight="bold", va="top")

    ax.set_xlabel("recall ของ grade 1 (mild) บนชุด test", fontsize=12, labelpad=10)
    ax.set_ylabel("จำนวนการทดลอง", fontsize=12, labelpad=10)
    ax.set_title(f"ทดลอง {len(vals)} รอบ ก็ยังกระจุกอยู่ที่เดิม — ชนเพดานข้อมูล ไม่ใช่เพดานวิธี",
                 fontsize=13.5, fontweight="bold", pad=16)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", alpha=0.22, linestyle=":")
    ax.set_axisbelow(True)

    # รันที่คะแนนสูงสุดมักเป็นรันที่ถูก early stopping ตัดตั้งแต่ epoch ต้นๆ ซึ่งตัวเลขใช้ไม่ได้
    # ต้องบอกไว้ด้วย ไม่งั้นจะดูเหมือนมีสูตรที่ทะลุเพดานได้จริง
    warn = (f"  ← หยุดที่ epoch {best[2]} ตัวเลขใช้ไม่ได้"
            if best[2] is not None and best[2] <= 3 else "")
    fig.text(0.5, -0.055,
             f"ชุด train มีปล้อง mild เพียง 224 ปล้อง · ชุด test มี {MILD_TEST_N} ปล้อง "
             f"(ทายถูกเพิ่ม 1 ใบ = +{1/MILD_TEST_N*100:.1f} pp)\n"
             f"รันที่คะแนนสูงสุด: {best[0]} = {best[1]:.3f}{warn}",
             ha="center", fontsize=10.5, color=ALERT if warn else GREY_D)
    _finish(fig, "fig09_mild_ceiling.png",
            "อ่านสดจาก outputs/runs/*/metrics.json (task = multiclass)")


def fig10_qc_flow():
    """สไลด์ 33 — QC gate: แยกความเสี่ยงจาก off-by-one ตามชนิดภาพ"""
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.set_xlim(0, 11); ax.set_ylim(0, 5.2); ax.axis("off")

    def box(x, y, w, h, text, fc, ec, fs=11, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06",
                                    fc=fc, ec=ec, lw=1.6))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
                fontweight="bold" if bold else "normal", color=INK)

    def arrow(x1, y1, x2, y2, col=GREY_D, label=None, ly=0):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=15, color=col, lw=1.7))
        if label:
            # พื้นขาวรองข้อความ กันไม่ให้เส้นลูกศรพาดทับตัวหนังสือ
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + ly, label, ha="center", va="center",
                    fontsize=9.8, color=col, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="none"))

    box(0.2, 2.1, 1.85, 1.0, "ภาพ DXA\nที่ยังไม่มี mask", "#eef2f3", "#c9d2d6")
    arrow(2.05, 2.6, 2.85, 2.6)
    box(2.85, 2.1, 1.9, 1.0, "U-Net\nauto-segment", "#e0f0f0", ACCENT, bold=True)
    arrow(4.75, 2.6, 5.5, 2.6)
    box(5.5, 2.1, 1.5, 1.0, "QC gate\nผ่าน ~90%", "#e0f0f0", ACCENT, bold=True)

    arrow(7.0, 2.9, 7.9, 4.0, ACCENT, "ภาพปกติ", 0.22)
    box(7.9, 3.6, 2.9, 0.95, "ใช้ได้เลย\nเลื่อนปล้องยังไงก็ยังเป็น grade 0",
        "#e0f0f0", ACCENT, fs=10.5)

    arrow(7.0, 2.3, 7.9, 1.2, WARN, "ภาพที่มีการหัก", -0.30)
    box(7.9, 0.75, 2.9, 0.95, "คนยืนยันตำแหน่ง\n(Label Studio)",
        "#f7ece2", WARN, fs=10.5, bold=True)

    ax.text(5.5, 4.72, "ปัญหาที่แท้จริงไม่ใช่ Dice แต่คือ “นับปล้องพลาด” — off-by-one 6.6%",
            ha="center", fontsize=13.5, fontweight="bold")
    ax.text(5.5, 0.28,
            "กติกา: ในขั้นยืนยันตำแหน่ง ห้ามอ่านเกรดใหม่เด็ดขาด — grade มาจาก DataTable เท่านั้น ยืนยันแค่ตำแหน่ง",
            ha="center", fontsize=10.5, color=WARN, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.5", fc="#f7ece2", ec=WARN, lw=1.1))
    _finish(fig, "fig10_qc_flow.png", "dl-sp/PIPELINE_masks_crop.md")


def fig11_dataset_growth():
    """สไลด์ 34 — ขยายข้อมูลได้ 2.2 เท่า แต่ที่โตขึ้นเกือบทั้งหมดคือ grade 0"""
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.4),
                             gridspec_kw={"width_ratios": [1.25, 1]})

    ax = axes[0]
    labels = list(DATASET_GROWTH)
    g0 = [DATASET_GROWTH[k]["g0"] for k in labels]
    fx = [DATASET_GROWTH[k]["total"] - DATASET_GROWTH[k]["g0"] for k in labels]
    b1 = ax.bar(labels, g0, color=GREY, width=0.52, label="grade 0 (ปกติ)")
    b2 = ax.bar(labels, fx, bottom=g0, color=ACCENT, width=0.52, label="grade 1–3 (หัก)")
    for i, k in enumerate(labels):
        tot = DATASET_GROWTH[k]["total"]
        ax.text(i, tot + 700, f"{tot:,} ปล้อง", ha="center", fontsize=12,
                fontweight="bold", color=INK)
        ax.text(i, g0[i] / 2, f"{g0[i]:,}\n({g0[i]/tot*100:.1f}%)", ha="center",
                va="center", fontsize=11, color="#4a5960")
        # แถบ grade 1-3 บางมาก วางตัวเลขไว้ข้างๆ แทนที่จะทับบนแถบ (ไม่งั้นชนป้ายยอดรวม)
        ax.annotate(f"{fx[i]:,} ปล้อง", xy=(i + 0.27, g0[i] + fx[i] / 2),
                    xytext=(i + 0.44, g0[i] + fx[i] / 2), ha="left", va="center",
                    fontsize=10.5, fontweight="bold", color=ACCENT,
                    arrowprops=dict(arrowstyle="-", color=ACCENT, lw=1.0))
    ax.set_ylim(0, 30500)
    ax.set_yticks([])
    ax.tick_params(axis="x", labelsize=11.5)
    ax.legend(fontsize=10.5, frameon=False, loc="upper left")
    ax.set_title("ขยายได้ 2.2 เท่า แต่ความไม่สมดุลแย่ลง", fontsize=13,
                 fontweight="bold", pad=12)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)

    ax = axes[1]
    ks = ["g0", "g1", "g2", "g3"]
    names = ["grade 0", "grade 1", "grade 2", "grade 3"]
    vals = [FRACTURE_PASS1[k] for k in ks]
    cols = [GREY, ALERT, GREY_D, GREY_D]
    bars = ax.bar(names, vals, color=cols, width=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 14, f"{v}", ha="center",
                fontsize=12, fontweight="bold",
                color=ALERT if b is bars[1] else INK)
    ax.set_ylim(0, 800)
    ax.set_yticks([])
    ax.set_title("รอบยืนยันเคสหักรอบแรก (56 คนไข้)", fontsize=13,
                 fontweight="bold", pad=12)
    ax.tick_params(axis="x", labelsize=11)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.text(1, 300, "ได้ mild เพิ่ม\nแค่ 23 ปล้อง", ha="center", fontsize=11.5,
            color=ALERT, fontweight="bold")

    fig.suptitle("ภาพที่ยังไม่มี mask ส่วนใหญ่เป็นเคสปกติ — การขยายจึงเติมแต่ grade 0",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.text(0.5, -0.075,
             "ประมาณการ: 2,067 ภาพที่เหลือ × หัก 19.7% ≈ 407 เคสที่หัก × 0.54 mild ต่อเคส "
             "≈ +220 ปล้อง → รวมราว 535 ปล้อง\n"
             "รอบแรกเดินไปแล้ว 56 จาก 407 เคส = 14%   (ตัวเลขประมาณการ ไม่ใช่ผลที่วัดแล้ว)",
             ha="center", fontsize=10.5, color=GREY_D)
    _finish(fig, "fig11_dataset_growth.png", "dl-sp/PIPELINE_masks_crop.md · fracture_manifest.csv")


def fig12_class_distribution():
    """สไลด์ 6 — ความไม่สมดุล 11:1 ให้เห็นด้วยตา"""
    labels = list(GRADE_COUNTS)
    vals = [GRADE_COUNTS[k] for k in labels]
    total = sum(vals)
    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    bars = ax.bar(labels, vals, color=[GREY, ALERT, GREY_D, GREY_D], width=0.58)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 190,
                f"{v:,}\n{v/total*100:.1f}%", ha="center", fontsize=11,
                fontweight="bold", color=INK)
    ax.set_ylim(0, 12800)
    ax.set_yticks([])
    ax.set_ylabel("จำนวนปล้องกระดูก", fontsize=12, labelpad=10)
    ax.tick_params(axis="x", labelsize=12)
    ax.set_title("ปล้องปกติ : ปล้องที่หัก ≈ 11 : 1   (ราย grade คือ 40 : 1)",
                 fontsize=14, fontweight="bold", pad=16)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    fig.text(0.5, -0.05,
             "ทายว่า “ปกติ” ทุกครั้งก็ได้ accuracy 91.7% แล้ว — ใช้ accuracy วัดผลไม่ได้ตั้งแต่ต้น",
             ha="center", fontsize=11.5, color=ALERT, fontweight="bold")
    _finish(fig, "fig12_class_distribution.png", "dl-sp/REPORT_distribution.md")


def fig13_fracture_by_level():
    """สไลด์ 7 — การหักกระจุกที่ T11/T12/L1"""
    levels = list(LEVEL_FX_PCT)[::-1]
    vals = [LEVEL_FX_PCT[k] for k in levels]
    ns = [LEVEL_FX_N[k] for k in levels]
    cols = [ACCENT if k in ("T11", "T12", "L1") else GREY for k in levels]

    fig, ax = plt.subplots(figsize=(8.8, 6.4))
    bars = ax.barh(levels, vals, color=cols, height=0.72)
    for b, v, n in zip(bars, vals, ns):
        ax.text(v + 0.45, b.get_y() + b.get_height() / 2, f"{v:.1f}%  ({n} ปล้อง)",
                va="center", fontsize=10.5, color=INK,
                fontweight="bold" if b.get_facecolor()[:3] == tuple(
                    int(ACCENT[i:i+2], 16) / 255 for i in (1, 3, 5)) else "normal")
    ax.set_xlim(0, 38)
    ax.set_xlabel("สัดส่วนปล้องที่หัก (%)", fontsize=12, labelpad=10)
    ax.set_title("T11 + T12 + L1 = 592 จาก 977 ปล้องที่หักทั้งหมด (60.6%)\nทั้งที่เป็นเพียง 3 ใน 15 ปล้อง",
                 fontsize=13.5, fontweight="bold", pad=16)
    ax.tick_params(axis="y", labelsize=11.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", alpha=0.22, linestyle=":")
    ax.set_axisbelow(True)
    fig.text(0.5, -0.045,
             "T3 มีปล้องที่หักเพียง 2 ปล้อง · T4 มี 1 ปล้อง — ช่วงบนไม่พอทั้งสำหรับเรียนรู้และประเมินผล",
             ha="center", fontsize=11, color=WARN, fontweight="bold")
    _finish(fig, "fig13_fracture_by_level.png", "dl-sp/REPORT_distribution.md · dist_per_vertebra.csv")


def fig14_offset_test():
    """สไลด์ 5 — ยืนยันว่า mask label 1-15 คือ T3-L5 จริง"""
    labels = list(OFFSET_TEST)
    vals = [OFFSET_TEST[k] for k in labels]
    cols = [GREY, GREY, ACCENT, GREY, GREY]
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    bars = ax.bar(labels, vals, color=cols, width=0.58)
    _barlabels(ax, bars, [f"{v:.3f}" for v in vals], INK, dy=0.008, fs=11.5)
    ax.set_ylim(0.7, 1.045)
    ax.set_xlabel("เลื่อน mapping ไปกี่ปล้อง", fontsize=12, labelpad=10)
    ax.set_ylabel("Jaccard กับปล้องที่แพทย์ประเมินได้", fontsize=11.5, labelpad=10)
    ax.set_title("offset 0 ชนะขาด — mask label 1–15 คือ T3–L5 จริง",
                 fontsize=13.5, fontweight="bold", pad=14)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", alpha=0.22, linestyle=":")
    ax.set_axisbelow(True)
    fig.text(0.5, -0.07,
             "ยืนยันด้วยหลักฐานอีก 2 ทางที่ไม่พึ่งกัน: label 15 อยู่ใน 832/833 ภาพและเป็นปล้องล่างสุดเสมอ\n"
             "และมี 9 ภาพที่ label ข้ามเลข = เลขเป็นตำแหน่งกายวิภาคตายตัว ไม่ใช่การนับเรียง",
             ha="center", fontsize=10.5, color=GREY_D)
    _finish(fig, "fig14_offset_test.png", "dl-sp/notes.txt [0]")


# ================================================================ main

FIGS = {
    1: fig01_shape_vs_texture,
    2: fig02_binary_tradeoff,
    3: fig03_metric_blindspot,
    4: fig04_selection_bias,
    5: fig05_genant_morphometry,
    6: fig06_onestage_vs_crop,
    7: fig07_fractures_per_patient,
    8: fig08_linear_probe,
    9: fig09_mild_ceiling,
    10: fig10_qc_flow,
    11: fig11_dataset_growth,
    12: fig12_class_distribution,
    13: fig13_fracture_by_level,
    14: fig14_offset_test,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", type=int, default=None,
                    help="เลขรูปที่ต้องการ (default = ทำทั้งหมด)")
    args = ap.parse_args()

    todo = args.only if args.only else sorted(FIGS)
    print(f"ฟอนต์ที่ใช้: {THAI_FONT}")
    print(f"เขียนลงที่: {OUT}\n")
    ok, fail = 0, []
    for k in todo:
        fn = FIGS.get(k)
        if fn is None:
            print(f"! ไม่มีรูปหมายเลข {k}")
            continue
        print(f"[{k:2d}] {fn.__doc__.strip().splitlines()[0]}")
        try:
            fn()
            ok += 1
        except Exception as e:            # ให้รูปที่เหลือทำต่อได้ ไม่ล้มทั้งชุด
            fail.append((k, repr(e)))
            print(f"  ! ล้ม: {e!r}")
    print(f"\nสำเร็จ {ok}/{len(todo)} รูป")
    if fail:
        print("รูปที่ล้ม:", ", ".join(str(k) for k, _ in fail))


if __name__ == "__main__":
    sys.stdout.reconfigure(newline=chr(10))
    main()
