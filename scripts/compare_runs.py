"""
compare_runs.py — รวมผลของทุกการทดลองมาเทียบกันในตารางเดียว (รันได้เลย ไม่ต้องเทรน)

input:  outputs/runs/*/metrics.json
            อ่านทุกการทดลองที่เคยรันไว้ (หรือเลือกเฉพาะที่ตรงกับ pattern ที่ระบุ)

output: outputs/comparison/comparison.csv
            ตารางเทียบ: 1 แถว = 1 การทดลอง มีทั้ง macro F1, AUC, recall ราย grade,
            best_epoch และการตั้งค่าที่ต่างกัน (preprocess/augment/resize/geometry)
        outputs/comparison/comparison_metrics.png
            กราฟแท่งเทียบ macro F1 กับ AUC ของทุกการทดลอง
        outputs/comparison/comparison_recall.png
            กราฟแท่งเทียบ recall ราย grade — สำคัญกว่าค่ารวม เพราะบอกว่าแต่ละวิธี
            ไปได้ไปเสียกับคลาสไหน (ค่ารวมกลบเรื่องนี้หมด)

ทำไมต้องมี: ค่า macro F1 ตัวเดียวเทียบข้ามการทดลองแล้วมักหลอก เพราะการทดลองที่
ดันคลาส normal ขึ้นนิดเดียวก็ได้คะแนนรวมดีขึ้นแล้ว ทั้งที่คลาสที่เราสนใจจริง
(mild) อาจแย่ลง — ต้องดู recall ราย grade ควบคู่เสมอ

USAGE:
    python scripts/compare_runs.py
    python scripts/compare_runs.py --pattern "diag_aug_*"
    python scripts/compare_runs.py --baseline effb0_masked_50ep15p_pp_baseline
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GRADES = ["normal", "mild", "moderate", "severe"]


def load_runs(runs_dir: Path, pattern: str) -> pd.DataFrame:
    """อ่าน metrics.json ของทุก run ที่ตรงกับ pattern แล้วรวมเป็นตารางเดียว"""
    rows = []
    for d in sorted(runs_dir.glob(pattern)):
        f = d / "metrics.json"
        if not f.exists():
            continue        # ข้าม run ที่ยังเทรนไม่จบ (ยังไม่มีไฟล์นี้)

        m = json.loads(f.read_text(encoding="utf-8"))
        row = {
            "run": d.name,
            "macro_f1": m["test_metrics"].get("macro_f1"),
            "auc": m["test_metrics"].get("auc"),
            "best_epoch": m.get("best_epoch"),
            # การตั้งค่าที่ต่างกันระหว่างการทดลอง — เก็บไว้ในตารางด้วยจะได้ไม่ต้อง
            # ย้อนไปเปิด config ทีละไฟล์ตอนอ่านผล
            "variant": m.get("variant"),
            "task": m.get("task"),
            "preprocess": m.get("preprocess", "?"),
            "augment": m.get("augment", "?"),
            "resize_mode": m.get("resize_mode", "?"),
            "metadata": m.get("use_metadata"),
            "geometry": m.get("use_geometry"),
        }
        for g in GRADES:
            info = m.get("per_grade_recall", {}).get(g)
            row[f"recall_{g}"] = info["recall"] if isinstance(info, dict) else info
        rows.append(row)

    if not rows:
        raise SystemExit(f"ไม่พบ metrics.json ใน {runs_dir}/{pattern} — ยังไม่มีการทดลองที่เทรนจบเลย")
    return pd.DataFrame(rows)


def add_delta(df: pd.DataFrame, baseline: str) -> pd.DataFrame:
    """
    เพิ่มคอลัมน์ส่วนต่างจาก baseline (หน่วย pp = percentage point)

    พร้อมตีความตามกฎในหัวข้อ 3.5 ของ experiment_summary.md:
      > 5 pp = น่าจะต่างจริง, 2-5 pp = สรุปไม่ได้, < 2 pp = ถือว่าเสมอ
    """
    if baseline not in set(df["run"]):
        print(f"  เตือน: ไม่พบ baseline ชื่อ '{baseline}' ในผลที่อ่านมา — ข้ามการคำนวณส่วนต่าง")
        return df

    base = df[df["run"] == baseline].iloc[0]
    for col in ["macro_f1", "auc"] + [f"recall_{g}" for g in GRADES]:
        df[f"d_{col}_pp"] = ((df[col] - base[col]) * 100).round(1)

    def verdict(pp):
        if pd.isna(pp):
            return ""
        a = abs(pp)
        if a < 2:
            return "เสมอ"
        if a <= 5:
            return "สรุปไม่ได้"
        return "ดีขึ้นจริง" if pp > 0 else "แย่ลงจริง"

    df["verdict_macro_f1"] = df["d_macro_f1_pp"].map(verdict)
    return df


def plot_metrics(df: pd.DataFrame, out_path: Path):
    """กราฟแท่งเทียบ macro F1 กับ AUC ของทุกการทดลอง"""
    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(max(7, 1.5 * len(df)), 4.2))
    ax.bar(x - 0.2, df["macro_f1"], width=0.4, label="macro F1", color="#4C78A8")
    ax.bar(x + 0.2, df["auc"], width=0.4, label="AUC", color="#F58518")
    for i, (f1, auc) in enumerate(zip(df["macro_f1"], df["auc"])):
        ax.text(i - 0.2, f1 + 0.012, f"{f1:.3f}", ha="center", fontsize=7)
        ax.text(i + 0.2, auc + 0.012, f"{auc:.3f}", ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(df["run"], rotation=30, ha="right", fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score (test set)")
    ax.set_title("Overall metrics by experiment")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_recall(df: pd.DataFrame, out_path: Path):
    """
    กราฟแท่งเทียบ recall ราย grade — กราฟที่สำคัญที่สุดของโปรเจกต์นี้

    เพราะข้อมูลไม่สมดุล 40:1 ค่ารวมจึงถูกคลาส normal ครอบงำ กราฟนี้เปิดให้เห็นว่า
    แต่ละวิธีแลกอะไรกับอะไร (เช่น ได้ severe มาแต่เสีย mild ไป)
    """
    cols = [f"recall_{g}" for g in GRADES]
    x = np.arange(len(GRADES))
    w = 0.8 / max(len(df), 1)

    fig, ax = plt.subplots(figsize=(max(7, 1.1 * len(df) + 4), 4.4))
    for i, (_, r) in enumerate(df.iterrows()):
        ax.bar(x + i * w - 0.4 + w / 2, [r[c] for c in cols], width=w, label=r["run"])
    ax.set_xticks(x)
    ax.set_xticklabels(GRADES)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("recall (test set)")
    ax.set_title("Per-grade recall by experiment")
    ax.legend(fontsize=7, ncol=1, bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", type=Path, default=Path("outputs/runs"))
    ap.add_argument("--pattern", default="*", help='เลือกเฉพาะบางการทดลอง เช่น "diag_aug_*"')
    ap.add_argument("--baseline", default="effb0_masked_50ep15p_pp_baseline",
                    help="ชื่อ run ที่ใช้เป็นตัวตั้งต้นสำหรับคำนวณส่วนต่าง")
    ap.add_argument("--out_dir", type=Path, default=Path("outputs/comparison"))
    args = ap.parse_args()

    df = load_runs(args.runs_dir, args.pattern)
    df = df.sort_values("macro_f1", ascending=False).reset_index(drop=True)
    df = add_delta(df, args.baseline)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_dir / "comparison.csv", index=False, encoding="utf-8-sig")
    plot_metrics(df, args.out_dir / "comparison_metrics.png")
    plot_recall(df, args.out_dir / "comparison_recall.png")

    show = ["run", "macro_f1", "auc", "best_epoch"] + [f"recall_{g}" for g in GRADES]
    if "d_macro_f1_pp" in df.columns:
        show += ["d_macro_f1_pp", "verdict_macro_f1"]
    print(f"\nเทียบ {len(df)} การทดลอง (เรียงตาม macro F1)\n")
    print(df[show].to_string(index=False))
    print(f"\nเซฟไว้ที่ {args.out_dir}")
    for f in sorted(args.out_dir.iterdir()):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
