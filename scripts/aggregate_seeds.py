"""
aggregate_seeds.py — รวมผลหลาย seed ของแต่ละเงื่อนไข แล้วรายงาน ค่าเฉลี่ย ± sd

ทำไมต้องมี: วัดแล้วพบว่า config เดียวกัน seed เดียวกัน รัน 2 ครั้งได้ macro F1
ต่างกัน 2.9 pp (0.567 vs 0.538) เพราะ dataloader หลาย worker กับ cuDNN ไม่
deterministic ถ้ารายงานตัวเลขจากรันเดียว จะแยกไม่ออกว่าความต่างที่เห็นเป็นผลจริง
หรือเป็นความผันผวนของการเทรน — ไฟล์นี้บังคับให้ทุกตัวเลขมีช่วงความไม่แน่นอนติดมาด้วย

เกณฑ์ตัดสินหลัก: **macro F1** ซึ่งเป็นตัวเดียวกับที่ใช้เลือก checkpoint ตอนเทรน
(select_metric ค่าเริ่มต้น) และเป็นตัวที่ตารางผลเดิมทั้งหมดรายงานไว้ จึงเทียบ
ย้อนหลังได้ตรงๆ

ข้อควรระวังที่ยังต้องดูควบคู่: คลาส normal มี 1,548 ปล้องจาก 1,680 (92%)
การขยับของ normal เพียง 1% จึงกลบผลของคลาสที่เราสนใจได้ — ไฟล์นี้จึงพิมพ์
recall ราย­คลาสไว้ใต้ macro F1 เสมอ และมีคอลัมน์ "เกิน noise?" กำกับทุกแถว
เพราะวัดแล้วพบว่า macro F1 ของ config เดียวกันแกว่งได้ถึง 2.9 pp ระหว่างรัน

USAGE:
    python scripts/aggregate_seeds.py
    python scripts/aggregate_seeds.py --prefix effb0_cmp_ --out outputs/comparison/SEEDS.md
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

CLASSES = ["normal", "mild", "moderate", "severe"]
# macro F1 มาก่อน (เกณฑ์ตัดสินหลัก ตัวเดียวกับที่ใช้เลือก checkpoint และกับตารางผลเดิม)
# ตามด้วย AUC และ recall ราย­คลาส ซึ่งเป็นตัวที่บอกว่าคะแนนรวมที่ได้มานั้นมาจากไหน
KEY_METRICS = [("macro_f1", "**macro F1**"), ("auc", "AUC"),
               ("mild_r", "mild recall"), ("moderate_r", "moderate recall"),
               ("severe_r", "severe recall"), ("normal_r", "normal recall"),
               ("mild_f1", "mild F1"), ("moderate_f1", "moderate F1")]


def load_run(d: Path) -> dict | None:
    mp = d / "metrics.json"
    if not mp.exists():
        return None
    m = json.load(open(mp, encoding="utf-8"))
    row = {"run": d.name, "best_epoch": m.get("best_epoch"),
           "macro_f1": m.get("test_metrics", {}).get("macro_f1"),
           "auc": m.get("test_metrics", {}).get("auc")}
    for c, v in m.get("per_grade_recall", {}).items():
        row[f"{c}_r"] = v.get("recall")
    per = d / "tables" / "metrics_overall.csv"
    if per.exists():
        t = pd.read_csv(per).set_index("class")
        for c in CLASSES:
            if c in t.index:
                row[f"{c}_p"] = t.loc[c, "precision"]
                row[f"{c}_f1"] = t.loc[c, "f1_score"]
    return row


def cond_of(name: str, prefix: str) -> str:
    """ตัด prefix และ _s<seed> ท้ายชื่อออก เหลือชื่อเงื่อนไข"""
    return re.sub(r"_s\d+$", "", name[len(prefix):] if name.startswith(prefix) else name)


def ms(vals) -> tuple:
    v = [x for x in vals if x is not None and not pd.isna(x)]
    if not v:
        return float("nan"), float("nan"), 0
    # ddof=1 = sd ของกลุ่มตัวอย่าง ไม่ใช่ของประชากร (เรามีแค่ 3 seed ไม่ใช่ทุก seed ที่เป็นไปได้)
    return float(np.mean(v)), (float(np.std(v, ddof=1)) if len(v) > 1 else 0.0), len(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", type=Path, default=Path("outputs/runs"))
    ap.add_argument("--prefix", default="effb0_cmp_")
    ap.add_argument("--out", type=Path, default=Path("outputs/comparison/SEEDS.md"))
    args = ap.parse_args()

    rows = [r for d in sorted(args.runs_dir.iterdir()) if d.is_dir() and d.name.startswith(args.prefix)
            for r in [load_run(d)] if r]
    if not rows:
        print(f"ไม่พบรันที่ขึ้นต้นด้วย '{args.prefix}' ที่รันจบแล้วเลย")
        return

    df = pd.DataFrame(rows)
    df["cond"] = df["run"].map(lambda n: cond_of(n, args.prefix))
    conds = sorted(df["cond"].unique())

    L = ["# เทียบ 2 เงื่อนไข — ค่าเฉลี่ย ± sd จากหลาย seed\n"]
    for c in conds:
        sub = df[df["cond"] == c]
        # คำนวณนอก f-string เพราะ f-string ใส่ backslash ในส่วน expression ไม่ได้
        # (Python < 3.12 เป็น SyntaxError — พังตอน import ไม่ใช่ตอนรัน จึงต้องระวัง)
        seeds = ", ".join(sub["run"].str.extract(r"_s(\d+)$")[0].fillna("?"))
        epochs = ", ".join(map(str, sub["best_epoch"]))
        L.append(f"- **{c}** — {len(sub)} seed: {seeds} (best epoch: {epochs})")
    L.append("")

    L.append("| ตัวชี้วัด | " + " | ".join(f"**{c}**" for c in conds) + " | ต่าง | เกิน noise? |")
    L.append("|---|" + "---|" * (len(conds) + 2))

    for key, label in KEY_METRICS:
        cells, stats = [], []
        for c in conds:
            m, s, n = ms(df[df["cond"] == c].get(key, []))
            stats.append((m, s, n))
            cells.append(f"{m:.3f} ± {s:.3f}" if n else "-")
        verdict = diff = "-"
        if len(stats) == 2 and all(st[2] for st in stats):
            (m1, s1, _), (m2, s2, _) = stats
            d = m2 - m1
            diff = f"{d:+.3f}"
            # เกณฑ์หยาบแต่ตรงไปตรงมา: ความต่างต้องมากกว่า sd รวมของทั้งสองกลุ่ม
            # ถึงจะเรียกว่าเห็นผล ไม่ใช่แค่ความผันผวนของการเทรน
            pooled = np.sqrt(s1 ** 2 + s2 ** 2)
            verdict = "**ใช่**" if abs(d) > pooled and pooled > 0 else "ไม่ (อยู่ในช่วง noise)"
        L.append(f"| {label} | " + " | ".join(cells) + f" | {diff} | {verdict} |")

    L.append("\n> คอลัมน์ 'ต่าง' = เงื่อนไขที่ 2 ลบเงื่อนไขที่ 1 · "
             "'เกิน noise' = |ต่าง| มากกว่า sd รวมของทั้งสองกลุ่มหรือไม่\n")

    L.append("\n## ตัวเลขรายรัน\n")
    cols = ["run", "best_epoch", "auc", "mild_r", "moderate_r", "severe_r", "normal_r", "macro_f1"]
    cols = [c for c in cols if c in df.columns]
    L.append("| " + " | ".join(cols) + " |")
    L.append("|" + "---|" * len(cols))
    for _, r in df.sort_values(["cond", "run"]).iterrows():
        L.append("| " + " | ".join(
            f"{r[c]:.3f}" if isinstance(r[c], float) else str(r[c]) for c in cols) + " |")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(L), encoding="utf-8")
    df.to_csv(args.out.with_suffix(".csv"), index=False)
    print(f"เขียนแล้ว: {args.out}\n           {args.out.with_suffix('.csv')}\n")
    print("\n".join(L[:6 + len(conds) + len(KEY_METRICS)]))


if __name__ == "__main__":
    main()
