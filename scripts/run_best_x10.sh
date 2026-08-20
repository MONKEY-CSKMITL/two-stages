#!/usr/bin/env bash
# run_best_x10.sh — เทียบ "ชุดเดิม vs ชุดขยาย 10 เท่า" ที่ค่าที่ดีที่สุดจาก sweep
#
# ลำดับ:
#   1. รอ sweep หลักจบ (มี GPU ตัวเดียว เริ่มทับกันแล้ว CUDA OOM เหมือนที่เคยพัง)
#   2. อ่านผู้ชนะ preprocess/augment จากผลจริงของ Part A/B
#   3. รันฝั่งชุดเดิมที่ค่านั้น 3 seed  <- ตัวคุม (คู่นี้ยังไม่เคยรัน เพราะ Part A
#      ตรึง augment=strong ส่วน Part B ตรึง preprocess=destripe)
#   4. สร้างชุด 10x ด้วย preprocess/augment ตัวเดียวกัน (คูณเฉพาะคลาส 1/2/3)
#   5. รันฝั่ง 10x 3 seed
#   6. ออกรายงาน
#
# USAGE: bash scripts/run_best_x10.sh

set -u
cd "$(dirname "$0")/.."
PY=./.venv/Scripts/python.exe
LOG=outputs/comparison/PROGRESS.log
export PYTHONIOENCODING=utf-8

run_one () {   # $1 = ชื่อรัน, $2 = ป้ายกำกับความคืบหน้า
  local name="$1" tag="$2"
  if [ -f "outputs/runs/${name}/metrics.json" ]; then
    echo "[$(date '+%H:%M:%S')] $tag ข้าม $name (มีผลแล้ว)" | tee -a "$LOG"; return
  fi
  local start=$(date +%s)
  $PY scripts/train.py --config "configs/stage2_${name}.yaml" > "outputs/comparison/log_${name}.txt" 2>&1
  local dur=$(( ($(date +%s) - start) / 60 ))
  if [ -f "outputs/runs/${name}/metrics.json" ]; then
    local res=$($PY -c "
import json
m=json.load(open('outputs/runs/${name}/metrics.json',encoding='utf-8'))
r=m['per_grade_recall']; t=m['test_metrics']
print(f\"F1={t['macro_f1']:.3f} AUC={t['auc']:.3f} mild={r['mild']['recall']:.3f} mod={r['moderate']['recall']:.3f} ep={m['best_epoch']}\")
" 2>/dev/null)
    echo "[$(date '+%H:%M:%S')] $tag OK   $name  ${dur}min  $res" | tee -a "$LOG"
  else
    echo "[$(date '+%H:%M:%S')] $tag FAIL $name  ${dur}min — ดู outputs/comparison/log_${name}.txt" | tee -a "$LOG"
  fi
}

echo "[$(date '+%H:%M:%S')] BEST-X10: รอ sweep หลักจบก่อน..." | tee -a "$LOG"
until grep -q "เสร็จทั้งหมด" "$LOG" 2>/dev/null; do sleep 120; done
# ยามชั้นสอง: ต้องไม่มี python เทรนค้างอยู่จริงๆ
while [ "$(powershell -NoProfile -Command '(Get-Process python -ErrorAction SilentlyContinue).Count' 2>/dev/null | tr -d '\r ')" -gt 0 ] 2>/dev/null; do sleep 60; done

echo "" | tee -a "$LOG"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] ===== BEST vs x10 =====" | tee -a "$LOG"
$PY scripts/make_x10_configs.py --stage winners | tee -a "$LOG"
eval "$($PY scripts/make_x10_configs.py --stage winners | grep '^BEST_')"
echo "[$(date '+%H:%M:%S')] ใช้ preprocess=$BEST_PP augment=$BEST_AUG" | tee -a "$LOG"

# --- สร้างชุด 10x ด้วยค่าเดียวกัน ---
echo "[$(date '+%H:%M:%S')] สร้างชุด 10x (preprocess=$BEST_PP augment=$BEST_AUG)..." | tee -a "$LOG"
$PY scripts/build_augmented_trainset.py --class_factors "0:1,1:10,2:10,3:10" \
    --preprocess "$BEST_PP" --augment "$BEST_AUG" --epochs 50 --patience 15 \
    > outputs/comparison/log_build_x10_best.txt 2>&1
grep -E "\[ok\]|แถว\)" outputs/comparison/log_build_x10_best.txt | tee -a "$LOG"

# สร้าง config ทั้งสองฝั่งไว้ก่อน (ไม่ได้ใช้ชื่อที่พิมพ์ออกมา จึงทิ้ง stdout)
$PY scripts/make_x10_configs.py --stage x10  > /dev/null
$PY scripts/make_x10_configs.py --stage orig > /dev/null

# --- สลับฟันปลาทีละ seed: 10x -> ชุดเดิม -> seed ถัดไป ---
# ถ้ารัน 10x ครบ 3 seed ก่อนแล้วค่อยรันตัวคุม จะต้องรอ 2.5 ชม. กว่าจะมีอะไร
# ให้เทียบสักคู่ — สลับแบบนี้ได้คู่แรกใน ~1.7 ชม. และหยุดกลางทางเมื่อไหร่
# ก็ยังได้ข้อมูลที่สมดุลเสมอ ไม่ใช่ได้ฝั่งเดียว
for s in 42 43 44; do
  run_one "effb0_best_x10_s${s}"  "(x10 s${s}) "
  run_one "effb0_best_orig_s${s}" "(orig s${s})"
  # ออกรายงานใหม่ทุกครั้งที่ครบ 1 คู่ จะได้เปิดดูระหว่างทางได้ ไม่ต้องรอจบทั้งหมด
  $PY scripts/aggregate_seeds.py --prefix effb0_best_ --out outputs/comparison/SEEDS_best_x10.md > /dev/null 2>&1
  echo "[$(date '+%H:%M:%S')] === ครบคู่ seed ${s} -> อัปเดต SEEDS_best_x10.md แล้ว ===" | tee -a "$LOG"
done

echo "[$(date '+%H:%M:%S')] ===== ทำรายงาน =====" | tee -a "$LOG"
$PY scripts/aggregate_seeds.py --prefix effb0_best_ --out outputs/comparison/SEEDS_best_x10.md > /dev/null 2>&1
$PY scripts/make_summary_report.py > /dev/null 2>&1
$PY scripts/compare_runs.py > /dev/null 2>&1
echo "[$(date '+%Y-%m-%d %H:%M:%S')] ===== BEST vs x10 เสร็จแล้ว =====" | tee -a "$LOG"
