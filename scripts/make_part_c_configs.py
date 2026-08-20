"""
make_part_c_configs.py — สร้าง config ของ Part C (กวาดน้ำหนักคลาส) จากผู้ชนะของ Part A/B

ทำไมต้องให้สคริปต์เลือกเอง ไม่ใช่ผมเลือกไว้ล่วงหน้า: ตอนตั้งคิว Part C ยังไม่รู้ว่า
preprocess/augment ตัวไหนชนะ ถ้าเดาไว้ก่อนแล้วเดาผิด จะได้ผลการกวาดน้ำหนักบนฐาน
ที่ไม่ใช่ตัวที่ดีที่สุด ซึ่งเป็นข้อบกพร่องเดียวกับรอบ WBOOST เมื่อวาน (กวาดน้ำหนัก
บนฐาน class_weights_from=source ที่พิสูจน์ทีหลังว่าเป็นทางตัน)

วิธีเลือกผู้ชนะ: ค่าเฉลี่ย macro F1 ข้าม seed ของแต่ละตัว (macro F1 คือเกณฑ์ที่
ตกลงกันไว้ และเป็นตัวเดียวกับที่ใช้เลือก checkpoint ตอนเทรน)

output: พิมพ์ชื่อรันทั้งหมดออก stdout บรรทัดละ 1 ชื่อ ให้ shell เอาไปวนรันต่อ

USAGE:
    python scripts/make_part_c_configs.py            # สร้าง config + พิมพ์ชื่อรัน
    python scripts/make_part_c_configs.py --dry_run  # ดูว่าจะเลือกอะไร ไม่เขียนไฟล์
"""

import argparse
import sys
import copy
import json
from pathlib import Path

import yaml

# scale ที่จะกวาด — [normal, mild, moderate, severe]
# [1,1,1,1] ต้องมีเสมอเพื่อเป็นตัวคุมของ Part C เอง เพราะคู่ (preprocess, augment)
# ที่ชนะอาจเป็นช่องที่ Part A/B ไม่เคยรัน (A ตรึง augment=strong, B ตรึง pp=destripe)
SCALES = {
    "base":     [1.0, 1.0, 1.0, 1.0],
    "m15d20":   [1.0, 1.5, 2.0, 1.0],
    "m20d25":   [1.0, 2.0, 2.5, 1.0],
    "modonly":  [1.0, 1.0, 2.0, 1.0],
}
SEEDS = (42, 43)


def mean_f1(runs_dir: Path, prefix: str) -> dict:
    """ค่าเฉลี่ย macro F1 ข้าม seed ของแต่ละค่าบนแกนนั้น"""
    acc = {}
    for d in runs_dir.iterdir():
        if not d.is_dir() or not d.name.startswith(prefix):
            continue
        mp = d / "metrics.json"
        if not mp.exists():
            continue
        # ตัด prefix ออก แล้วตัด _s<seed> ท้ายออก เหลือชื่อค่าบนแกน
        val = d.name[len(prefix):].rsplit("_s", 1)[0]
        f1 = json.load(open(mp, encoding="utf-8"))["test_metrics"]["macro_f1"]
        acc.setdefault(val, []).append(f1)
    return {k: sum(v) / len(v) for k, v in acc.items() if v}


def main():
    # Windows พิมพ์ตัวขึ้นบรรทัดเป็น CRLF ทำให้ชื่อรันที่ shell รับไปมี CR ติดท้าย
    sys.stdout.reconfigure(newline=chr(10))
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", type=Path, default=Path("outputs/runs"))
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    pp_scores = mean_f1(args.runs_dir, "effb0_sw_pp_")
    aug_scores = mean_f1(args.runs_dir, "effb0_sw_aug_")

    if not pp_scores or not aug_scores:
        raise SystemExit("ยังไม่มีผลของ Part A/B พอให้เลือกผู้ชนะ — ต้องรอ sweep จบก่อน")

    best_pp = max(pp_scores, key=pp_scores.get)
    best_aug = max(aug_scores, key=aug_scores.get)

    print(f"# ผู้ชนะ preprocess: {best_pp}  (macro F1 เฉลี่ย {pp_scores[best_pp]:.4f})", flush=True)
    for k in sorted(pp_scores, key=pp_scores.get, reverse=True):
        print(f"#     {k:<20} {pp_scores[k]:.4f}", flush=True)
    print(f"# ผู้ชนะ augment: {best_aug}  (macro F1 เฉลี่ย {aug_scores[best_aug]:.4f})", flush=True)
    for k in sorted(aug_scores, key=aug_scores.get, reverse=True):
        print(f"#     {k:<20} {aug_scores[k]:.4f}", flush=True)

    BASE = {
        "experiment": {"name": None, "seed": None},
        "data": {"split_dir": "data/processed/splits", "variant": "xray_masked",
                 "task": "multiclass", "img_size": None, "resize_mode": "pad",
                 "preprocess": best_pp, "augment": best_aug},
        "model": {"backbone": "efficientnet_b0", "pretrained": True, "level_embed_dim": 32,
                  "head_dropout": 0.2, "freeze_backbone": False},
        "loss": {"gamma": 2.0, "use_class_weights": True, "class_weight_scale": None},
        "train": {"epochs": 50, "batch_size": 64, "lr": 1.0e-4, "weight_decay": 1.0e-4,
                  "patience": 15, "num_workers": 4},
        "output": {"dir": None},
    }
    head = (f"# Part C — กวาดน้ำหนักคลาส บนฐานที่ชนะจาก Part A/B\n"
            f"# preprocess={best_pp} (macro F1 เฉลี่ย {pp_scores[best_pp]:.4f})\n"
            f"# augment={best_aug} (macro F1 เฉลี่ย {aug_scores[best_aug]:.4f})\n"
            f"# class_weight_scale คูณทับน้ำหนักอัตโนมัติ [normal, mild, moderate, severe]\n"
            f"# แถว base [1,1,1,1] คือตัวคุม ต้องมีเพราะคู่ (preprocess, augment) นี้\n"
            f"# อาจเป็นช่องที่ Part A/B ไม่เคยรัน (A ตรึง augment=strong, B ตรึง pp=destripe)\n")

    names = []
    for tag, scale in SCALES.items():
        for seed in SEEDS:
            name = f"effb0_pc_w{tag}_s{seed}"
            names.append(name)
            if args.dry_run:
                continue
            c = copy.deepcopy(BASE)
            c["experiment"].update(name=name, seed=seed)
            c["loss"]["class_weight_scale"] = scale
            c["output"]["dir"] = f"outputs/runs/{name}"
            Path(f"configs/stage2_{name}.yaml").write_text(
                head + "\n" + yaml.safe_dump(c, allow_unicode=True, sort_keys=False),
                encoding="utf-8")

    print(f"# รวม {len(names)} รัน ({len(SCALES)} scale x {len(SEEDS)} seed)", flush=True)
    for n in names:
        print(n, flush=True)


if __name__ == "__main__":
    main()
