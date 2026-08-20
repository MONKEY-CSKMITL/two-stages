"""
make_x10_configs.py — สร้าง config เทียบ "ชุดเดิม vs ชุดขยาย 10 เท่า" ที่ค่าที่ดีที่สุด

ใช้ผู้ชนะจริงจาก sweep (Part A/B) ไม่ใช่เดาไว้ล่วงหน้า เพราะตอนตั้งคิวยังไม่รู้ผล

--------------------------------------------------------------------------
ทำไมต้องรันฝั่ง "ชุดเดิม" ใหม่ด้วย ทั้งที่ sweep รันไปแล้ว
--------------------------------------------------------------------------
Part A ตรึง augment=strong ส่วน Part B ตรึง preprocess=destripe
คู่ (preprocess ที่ชนะ, augment ที่ชนะ) จึงเป็น "ช่องที่ตัดกัน" ซึ่งยังไม่เคยรันเลย
ถ้าไม่รันฝั่งชุดเดิมที่ค่านี้ จะไม่มีตัวคุมให้ฝั่ง 10x เทียบ

--------------------------------------------------------------------------
ทำไมฝั่ง 10x ยังเปิด augment สุ่มสดทับอีกชั้น
--------------------------------------------------------------------------
วัดมาแล้วว่าชุด offline ที่ปิด augment ถูกจำได้หมดตั้งแต่ epoch 3 (train_loss
0.297 -> 0.025) เพราะ 1 ปล้องมีแค่ 10 หน้าตาตายตัว การเปิด augment ทับทำให้ได้
ทั้งไฟล์ 10 เท่าที่นับได้ และหน้าตาใหม่ทุก epoch ที่จำไม่ได้ — และยังทำให้คลาส
normal (ที่ไม่ได้คูณ) โดน augment ด้วย จึงลบทางลัด "ภาพที่ถูกดัดแปลง = ไม่ปกติ"
ที่การคูณเฉพาะคลาสน้อยสร้างขึ้นโดยไม่ตั้งใจ

USAGE:
    python scripts/make_x10_configs.py --stage orig   # สร้าง config ฝั่งชุดเดิม
    python scripts/make_x10_configs.py --stage x10    # สร้าง config ฝั่ง 10x (ต้องสร้างชุดข้อมูลก่อน)
    python scripts/make_x10_configs.py --stage winners  # พิมพ์ผู้ชนะอย่างเดียว
"""

import argparse
import sys
import copy
import json
from pathlib import Path

import yaml

SEEDS = (42, 43, 44)


def mean_f1(runs_dir: Path, prefix: str) -> dict:
    acc = {}
    for d in runs_dir.iterdir():
        if not d.is_dir() or not d.name.startswith(prefix):
            continue
        mp = d / "metrics.json"
        if not mp.exists():
            continue
        val = d.name[len(prefix):].rsplit("_s", 1)[0]
        acc.setdefault(val, []).append(
            json.load(open(mp, encoding="utf-8"))["test_metrics"]["macro_f1"])
    return {k: sum(v) / len(v) for k, v in acc.items() if v}


def pick(runs_dir: Path):
    pp = mean_f1(runs_dir, "effb0_sw_pp_")
    aug = mean_f1(runs_dir, "effb0_sw_aug_")
    if not pp or not aug:
        raise SystemExit("ยังไม่มีผล sweep พอให้เลือกผู้ชนะ")
    return max(pp, key=pp.get), max(aug, key=aug.get), pp, aug


def write(name, variant, preprocess, augment, seed):
    cfg = {
        "experiment": {"name": name, "seed": seed},
        "data": {"split_dir": "data/processed/splits", "variant": variant,
                 "task": "multiclass", "img_size": None, "resize_mode": "pad",
                 "preprocess": preprocess, "augment": augment},
        "model": {"backbone": "efficientnet_b0", "pretrained": True, "level_embed_dim": 32,
                  "head_dropout": 0.2, "freeze_backbone": False},
        "loss": {"gamma": 2.0, "use_class_weights": True},
        "train": {"epochs": 50, "batch_size": 64, "lr": 1.0e-4, "weight_decay": 1.0e-4,
                  "patience": 15, "num_workers": 4},
        "output": {"dir": f"outputs/runs/{name}"},
    }
    head = ("# เทียบชุดเดิม vs ชุด 10x ที่ค่าที่ดีที่สุดจาก sweep\n"
            f"# preprocess={preprocess} · augment={augment} · 50ep/p15 · 3 seed\n"
            "# ทั้งสองฝั่งใช้ augment สุ่มสดเหมือนกัน ต่างกันแค่ชุดข้อมูล\n")
    Path(f"configs/stage2_{name}.yaml").write_text(
        head + "\n" + yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")


def main():
    # Windows พิมพ์ตัวขึ้นบรรทัดเป็น CRLF ทำให้ชื่อรันที่ shell รับไปมี CR
    # ติดท้าย แล้วหา config ไม่เจอ (พังมาแล้วจริง ล้ม 3 รันรวด)
    sys.stdout.reconfigure(newline=chr(10))
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", type=Path, default=Path("outputs/runs"))
    ap.add_argument("--stage", choices=["winners", "orig", "x10"], required=True)
    args = ap.parse_args()

    best_pp, best_aug, pp, aug = pick(args.runs_dir)

    if args.stage == "winners":
        print(f"BEST_PP={best_pp}")
        print(f"BEST_AUG={best_aug}")
        print(f"# preprocess: " + "  ".join(f"{k}={v:.4f}" for k, v in
                                            sorted(pp.items(), key=lambda x: -x[1])[:4]))
        print(f"# augment: " + "  ".join(f"{k}={v:.4f}" for k, v in
                                         sorted(aug.items(), key=lambda x: -x[1])[:4]))
        return

    if args.stage == "orig":
        # ฝั่งชุดเดิม: preprocess ทำสดตอนโหลด augment สุ่มสด
        for s in SEEDS:
            write(f"effb0_best_orig_s{s}", "xray_masked", best_pp, best_aug, s)
            print(f"effb0_best_orig_s{s}")
    else:
        # ฝั่ง 10x: preprocess ถูก bake ลงไฟล์แล้วตอนสร้างชุด จึงตั้งเป็น none
        # ส่วน augment ยังเปิดสุ่มสดทับอีกชั้น (เหตุผลอยู่ในหัวไฟล์)
        variant = f"xray_masked_{best_pp}_{best_aug}_cls1-10-10-10"
        for s in SEEDS:
            write(f"effb0_best_x10_s{s}", variant, "none", best_aug, s)
            print(f"effb0_best_x10_s{s}")


if __name__ == "__main__":
    main()
