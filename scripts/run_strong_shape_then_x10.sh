#!/usr/bin/env bash
# run_strong_shape_then_x10.sh — แทรก strong_shape ก่อน แล้วค่อยทำ x10
#
# ทำไมต้องแทรกก่อน: ตอนนี้ strong นำ macro F1 (0.614) ส่วน standard_shape นำ
# mild/moderate (0.457/0.627) และการเติม shape ให้ standard เพิ่มได้ +3.5 pp
# ถ้า strong_shape ชนะจริง แล้วเราสร้างชุด 10x ไปก่อนโดยไม่รู้ จะต้องรื้อสร้างใหม่
# + รันใหม่อีก 3 ชม. ซึ่งแพงกว่าการรัน strong_shape 2 seed (~1.5 ชม.) ตอนนี้
#
# USAGE: bash scripts/run_strong_shape_then_x10.sh

set -u
cd "$(dirname "$0")/.."
PY=./.venv/Scripts/python.exe
LOG=outputs/comparison/PROGRESS.log
export PYTHONIOENCODING=utf-8

echo "[$(date '+%H:%M:%S')] STRONG_SHAPE: รอ sweep หลักจบก่อน..." | tee -a "$LOG"
until grep -q "เสร็จทั้งหมด" "$LOG" 2>/dev/null; do sleep 120; done
while [ "$(powershell -NoProfile -Command '(Get-Process python -ErrorAction SilentlyContinue).Count' 2>/dev/null | tr -d '\r ')" -gt 0 ] 2>/dev/null; do sleep 60; done

echo "" | tee -a "$LOG"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] ===== strong_shape (2 seed) =====" | tee -a "$LOG"

for name in effb0_sw_aug_strong_shape_s42 effb0_sw_aug_strong_shape_s43; do
  if [ -f "outputs/runs/${name}/metrics.json" ]; then
    echo "[$(date '+%H:%M:%S')] ข้าม $name" | tee -a "$LOG"; continue
  fi
  start=$(date +%s)
  $PY scripts/train.py --config "configs/stage2_${name}.yaml" > "outputs/comparison/log_${name}.txt" 2>&1
  dur=$(( ($(date +%s) - start) / 60 ))
  if [ -f "outputs/runs/${name}/metrics.json" ]; then
    res=$($PY -c "
import json
m=json.load(open('outputs/runs/${name}/metrics.json',encoding='utf-8'))
r=m['per_grade_recall']; t=m['test_metrics']
print(f\"F1={t['macro_f1']:.3f} AUC={t['auc']:.3f} mild={r['mild']['recall']:.3f} mod={r['moderate']['recall']:.3f} ep={m['best_epoch']}\")
" 2>/dev/null)
    echo "[$(date '+%H:%M:%S')] OK   $name  ${dur}min  $res" | tee -a "$LOG"
  else
    echo "[$(date '+%H:%M:%S')] FAIL $name  ${dur}min" | tee -a "$LOG"
  fi
done

$PY scripts/aggregate_seeds.py --prefix effb0_sw_aug_ --out outputs/comparison/SEEDS_augment.md > /dev/null 2>&1
echo "[$(date '+%H:%M:%S')] strong_shape จบ -> ต่อด้วย x10 (จะเลือกผู้ชนะใหม่จากผลล่าสุด)" | tee -a "$LOG"

# เรียกต่อในโปรเซสเดียวกัน (exec) เพื่อไม่ให้เหลือ bash ค้างซ้อนกันเป็นผี
exec bash scripts/run_best_x10.sh
