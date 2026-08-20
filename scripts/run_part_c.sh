#!/usr/bin/env bash
# run_part_c.sh — กวาดน้ำหนักคลาส ต่อจาก Part A/B
#
# รอ run_all_sweeps.sh จบก่อนเสมอ (ดูจากบรรทัด "เสร็จทั้งหมด" ใน PROGRESS.log)
# เพราะมี GPU ตัวเดียว ถ้าเริ่มทับกันจะ CUDA out of memory เหมือนที่เคยพังมาแล้ว
#
# แล้วให้ make_part_c_configs.py เลือกผู้ชนะ preprocess/augment จากผลจริง
# ไม่ใช่เดาไว้ล่วงหน้า — จะได้กวาดน้ำหนักบนฐานที่ดีที่สุดจริง
#
# USAGE: bash scripts/run_part_c.sh

set -u
cd "$(dirname "$0")/.."
PY=./.venv/Scripts/python.exe
LOG=outputs/comparison/PROGRESS.log
export PYTHONIOENCODING=utf-8

echo "[$(date '+%H:%M:%S')] Part C: รอ sweep หลักจบก่อน..." | tee -a "$LOG"
until grep -q "เสร็จทั้งหมด" "$LOG" 2>/dev/null; do sleep 120; done

# กันชนซ้ำอีกชั้น: ต้องไม่มี python เทรนค้างอยู่จริงๆ ก่อนเริ่ม
while [ "$(powershell -NoProfile -Command '(Get-Process python -ErrorAction SilentlyContinue).Count' 2>/dev/null | tr -d '\r ')" -gt 0 ] 2>/dev/null; do
  sleep 60
done

echo "" | tee -a "$LOG"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] ===== เริ่ม Part C (กวาดน้ำหนักคลาส) =====" | tee -a "$LOG"

# สร้าง config แล้วรับรายชื่อรันกลับมา (บรรทัดที่ขึ้นต้นด้วย # เป็นข้อมูลสรุป ไม่ใช่ชื่อรัน)
OUT=$($PY scripts/make_part_c_configs.py 2>&1)
echo "$OUT" | grep '^#' | tee -a "$LOG"
RUNS=$(echo "$OUT" | grep -v '^#')

set -- $RUNS
total=$#
i=0
t0=$(date +%s)

for name in "$@"; do
  i=$((i+1))
  cfg="configs/stage2_${name}.yaml"
  if [ -f "outputs/runs/${name}/metrics.json" ]; then
    echo "[$(date '+%H:%M:%S')] (C $i/$total) ข้าม $name" | tee -a "$LOG"; continue
  fi
  start=$(date +%s)
  $PY scripts/train.py --config "$cfg" > "outputs/comparison/log_${name}.txt" 2>&1
  dur=$(( ($(date +%s) - start) / 60 ))
  if [ -f "outputs/runs/${name}/metrics.json" ]; then
    res=$($PY -c "
import json
m=json.load(open('outputs/runs/${name}/metrics.json',encoding='utf-8'))
r=m['per_grade_recall']; t=m['test_metrics']
print(f\"F1={t['macro_f1']:.3f} AUC={t['auc']:.3f} mild={r['mild']['recall']:.3f} mod={r['moderate']['recall']:.3f} ep={m['best_epoch']}\")
" 2>/dev/null); status="OK  "
  else
    res="— ดู outputs/comparison/log_${name}.txt"; status="FAIL"
  fi
  el=$(( ($(date +%s) - t0) / 60 )); eta=$(( el * (total - i) / i ))
  echo "[$(date '+%H:%M:%S')] (C $i/$total) $status $name  ${dur}min  $res  | เหลือ ~${eta} นาที" | tee -a "$LOG"
done

echo "[$(date '+%H:%M:%S')] ===== Part C จบ กำลังทำรายงาน =====" | tee -a "$LOG"
$PY scripts/aggregate_seeds.py --prefix effb0_pc_w --out outputs/comparison/SEEDS_weights.md > /dev/null 2>&1
$PY scripts/make_summary_report.py > /dev/null 2>&1
$PY scripts/compare_runs.py > /dev/null 2>&1
echo "[$(date '+%Y-%m-%d %H:%M:%S')] ===== เสร็จทุกอย่างแล้ว =====" | tee -a "$LOG"
