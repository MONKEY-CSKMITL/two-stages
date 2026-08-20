#!/usr/bin/env bash
# run_x10_weights.sh — ชุด 10x + ดันน้ำหนักคลาส mild/moderate
#
# ที่มา: สูตรน้ำหนัก w_c = N/(K*n_c) ทำให้ n_c * w_c = N/K เท่ากันทุกคลาสเสมอ
# การคูณข้อมูลคลาสน้อยขึ้น 10 เท่าจึงถูกหักล้างทิ้งหมด (mild: 224x9.39 = 2102
# กลายเป็น 2240x1.65 = 3689 แต่ยังเท่ากับคลาสอื่นเป๊ะเหมือนเดิม) = สมดุลที่โมเดล
# เห็นไม่เปลี่ยนเลย ต้องดันน้ำหนักเองถึงจะขยับจุดตัดได้จริง
#
# หลักฐานหนุน: ชุด 10x ชนะ AUC ทั้ง 2 seed (0.940/0.935 vs 0.920) แต่แพ้ macro F1
# AUC ไม่ขึ้นกับจุดตัด ส่วน macro F1 ขึ้น -> ปัญหาอยู่ที่จุดตัด ไม่ใช่ที่ข้อมูล
#
# USAGE: bash scripts/run_x10_weights.sh

set -u
cd "$(dirname "$0")/.."
PY=./.venv/Scripts/python.exe
LOG=outputs/comparison/PROGRESS.log
export PYTHONIOENCODING=utf-8

echo "[$(date '+%H:%M:%S')] X10-WEIGHTS: รอคิวก่อนหน้าจบ..." | tee -a "$LOG"
until grep -q "BEST vs x10 เสร็จแล้ว" "$LOG" 2>/dev/null; do sleep 120; done
while [ "$(powershell -NoProfile -Command '(Get-Process python -ErrorAction SilentlyContinue).Count' 2>/dev/null | tr -d '\r ')" -gt 0 ] 2>/dev/null; do sleep 60; done

echo "" | tee -a "$LOG"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] ===== ชุด 10x + ดันน้ำหนักคลาส =====" | tee -a "$LOG"

# สลับ tag ทีละ seed เพื่อให้ได้คู่ที่เทียบกันได้เร็วที่สุด และหยุดกลางทางได้เสมอ
for s in 42 43; do
for tag in m15d20 modonly; do
  name="effb0_x10w_${tag}_s${s}"
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
    echo "[$(date '+%H:%M:%S')] FAIL $name  ${dur}min — ดู outputs/comparison/log_${name}.txt" | tee -a "$LOG"
  fi
  $PY scripts/aggregate_seeds.py --prefix effb0_x10w_ --out outputs/comparison/SEEDS_x10_weights.md > /dev/null 2>&1
done
done

$PY scripts/make_summary_report.py > /dev/null 2>&1
echo "[$(date '+%Y-%m-%d %H:%M:%S')] ===== 10x + น้ำหนัก เสร็จแล้ว =====" | tee -a "$LOG"
