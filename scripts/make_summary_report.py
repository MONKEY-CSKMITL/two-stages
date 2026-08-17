"""
make_summary_report.py — รวมผลทุกการทดลองเป็นรายงาน markdown อ่านครั้งเดียวจบ

ต่างจาก compare_runs.py: ตัวนั้นทำ csv + กราฟไว้เปิดดูเอง ส่วนไฟล์นี้ทำ "รายงาน
ที่อ่านแล้วเข้าใจเลย" มี precision/recall/f1 ราย**คลาส**ครบทุกรัน ซึ่ง metrics.json
ไม่มี (มีแต่ recall) ต้องไปดึงจาก tables/metrics_overall.csv ของแต่ละรัน

USAGE:
    python scripts/make_summary_report.py
    python scripts/make_summary_report.py --out outputs/comparison/SUMMARY.md
"""

import argparse
import json
from pathlib import Path

import pandas as pd

CLASSES = ["normal", "mild", "moderate", "severe"]

# รันอ้างอิงที่ใช้เป็นหมุดเทียบ — ตัวเลขเหล่านี้คือสถิติเดิมก่อนเริ่มรอบนี้
REFERENCE = "effb0_masked_15ep_ds_destripe"


def load_all(runs_dir: Path) -> list:
    """อ่านทุกรันที่มี metrics.json — รันที่ยังไม่จบจะไม่มีไฟล์นี้ จึงถูกข้ามไปเอง"""
    out = []
    for d in sorted(runs_dir.iterdir()):
        mp = d / "metrics.json"
        if not mp.exists():
            continue
        m = json.load(open(mp, encoding="utf-8"))
        row = {
            "run": d.name,
            "variant": m.get("variant", ""),
            "preprocess": m.get("preprocess", ""),
            "augment": m.get("augment", ""),
            "best_epoch": m.get("best_epoch"),
            "macro_f1": m.get("test_metrics", {}).get("macro_f1"),
            "auc": m.get("test_metrics", {}).get("auc"),
        }
        per = d / "tables" / "metrics_overall.csv"
        if per.exists():
            t = pd.read_csv(per).set_index("class")
            for c in CLASSES:
                if c in t.index:
                    row[f"{c}_p"] = t.loc[c, "precision"]
                    row[f"{c}_r"] = t.loc[c, "recall"]
                    row[f"{c}_f1"] = t.loc[c, "f1_score"]
        out.append(row)
    return out


def fmt(v, nd=3):
    return "-" if v is None or pd.isna(v) else f"{v:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", type=Path, default=Path("outputs/runs"))
    ap.add_argument("--out", type=Path, default=Path("outputs/comparison/SUMMARY.md"))
    args = ap.parse_args()

    rows = load_all(args.runs_dir)
    if not rows:
        print("ไม่พบรันที่มี metrics.json เลย")
        return

    df = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    L = []
    L.append("# สรุปผลการทดลองทั้งหมด\n")
    L.append(f"อ่านจาก `{args.runs_dir}` — {len(df)} การทดลองที่รันจบแล้ว\n")

    ref = df[df["run"] == REFERENCE]
    if len(ref):
        r = ref.iloc[0]
        L.append(f"> **หมุดเทียบ** `{REFERENCE}` — macro F1 {fmt(r['macro_f1'])} · "
                 f"AUC {fmt(r['auc'])} · mild recall **{fmt(r.get('mild_r'))}**\n")

    # --- ตารางรวม เรียงตาม macro F1 ---
    L.append("\n## ภาพรวม (เรียงตาม macro F1)\n")
    L.append("| การทดลอง | aug | ep | macro F1 | AUC | recall: ปกติ | เล็กน้อย | ปานกลาง | รุนแรง |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for _, r in df.iterrows():
        L.append(f"| `{r['run']}` | {r['augment']} | {r['best_epoch']} | "
                 f"{fmt(r['macro_f1'])} | {fmt(r['auc'])} | {fmt(r.get('normal_r'))} | "
                 f"**{fmt(r.get('mild_r'))}** | {fmt(r.get('moderate_r'))} | {fmt(r.get('severe_r'))} |")

    # --- ตารางรายคลาสเต็ม precision / recall / f1 ---
    L.append("\n## รายคลาสครบ precision / recall / F1\n")
    for _, r in df.iterrows():
        L.append(f"\n### `{r['run']}`")
        L.append(f"variant={r['variant']} · preprocess={r['preprocess']} · augment={r['augment']} · "
                 f"best_epoch={r['best_epoch']} · macro F1 {fmt(r['macro_f1'])} · AUC {fmt(r['auc'])}\n")
        L.append("| คลาส | precision | recall | f1 |")
        L.append("|---|---|---|---|")
        for c in CLASSES:
            L.append(f"| {c} | {fmt(r.get(f'{c}_p'))} | {fmt(r.get(f'{c}_r'))} | {fmt(r.get(f'{c}_f1'))} |")
        # macro average คำนวณเองจากรายคลาส (metrics.json เก็บแต่ macro f1 ไม่มี macro p/r)
        for stat, lab in [("p", "macro precision"), ("r", "macro recall"), ("f1", "macro f1")]:
            vals = [r.get(f"{c}_{stat}") for c in CLASSES]
            vals = [v for v in vals if v is not None and not pd.isna(v)]
            if vals:
                L.append(f"| **{lab}** | | | {fmt(sum(vals) / len(vals))} |")

    args.out.write_text("\n".join(L), encoding="utf-8")
    df.to_csv(args.out.with_suffix(".csv"), index=False)
    print(f"เขียนรายงานแล้ว: {args.out}")
    print(f"                 {args.out.with_suffix('.csv')}")
    print(f"\n3 อันดับแรกตาม macro F1:")
    for _, r in df.head(3).iterrows():
        print(f"  {r['run'][:48]:<50} F1={fmt(r['macro_f1'])} mild={fmt(r.get('mild_r'))}")


if __name__ == "__main__":
    main()
