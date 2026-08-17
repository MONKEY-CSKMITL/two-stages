#!/usr/bin/env bash
# run_all_sweeps.sh — ตัวขับงานทั้งหมดของรอบนี้ "สายเดียว โปรเซสเดียว"
#
# --- ทำไมต้องรวมเป็นสายเดียว ---
# ครั้งก่อนแยกเป็นหลาย batch แล้วใช้ lock กันชนกัน ปรากฏว่าพังเพราะ 2 เหตุ:
#   1. คิวชุดแรกไม่ได้ใช้ lock เลย -> batch อื่นคว้า lock ไปรันทับ = เทรน 2 ตัว
#      พร้อมกันบน GPU 8 GB -> CUDA out of memory และ paging file เต็ม
#   2. พอสั่งหยุด task ตัวแม่ สคริปต์ลูกยังรันต่อเป็นผี -> ยิ่งรันทับกันหนักขึ้น
# สายเดียวไม่มีทางชนตัวเองได้ ไม่ต้องมี lock ไม่ต้องมีตัวรอ และหยุดทีเดียวจบ
#
# --- ล้มแล้วไปต่อ / รันซ้ำได้ ---
# รันไหนพังบันทึก FAIL แล้วขึ้นตัวถัดไป ไม่หยุดทั้งสาย
# รันไหนมี metrics.json แล้วจะถูกข้าม -> เครื่องดับกลางคืนก็สั่งซ้ำได้ ไม่เริ่มใหม่หมด
#
# USAGE: bash scripts/run_all_sweeps.sh

set -u
cd "$(dirname "$0")/.."
PY=./.venv/Scripts/python.exe
LOG=outputs/comparison/PROGRESS.log
mkdir -p outputs/comparison
export PYTHONIOENCODING=utf-8
# ลดการแตกกระจายของหน่วยความจำ GPU — กันกรณี OOM ตอนโมเดลขอบล็อกเล็กๆ ต่อเนื่อง
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PPL="none normalize clahe normalize_clahe destripe destripe_clahe unsharp flatten destripe_unsharp destripe_flatten"
AUG="none intensity geometric standard strong shape standard_shape"

RUNS=""
# ลำดับสำคัญ: เอาการเปรียบเทียบหลัก (ชุดเดิม vs ชุด 10x) ขึ้นก่อน
# เพราะเป็นคำถามหลักของรอบนี้ ถ้าต้องหยุดกลางทางจะได้มีคำตอบข้อนี้ก่อนเสมอ
# และไล่ทีละ seed สลับเงื่อนไข เพื่อให้ได้ "คู่ที่เทียบกันได้" ตั้งแต่ seed แรก
for s in 42 43 44; do for c in orig off10x; do RUNS="$RUNS effb0_cmp_${c}_50ep15p_s${s}"; done; done
for s in 42 43; do for p in $PPL; do RUNS="$RUNS effb0_sw_pp_${p}_s${s}"; done; done
for s in 42 43; do for a in $AUG; do RUNS="$RUNS effb0_sw_aug_${a}_s${s}"; done; done

set -- $RUNS
total=$#
i=0
t0=$(date +%s)
echo "" | tee -a "$LOG"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] ===== เริ่มสายเดียว $total รัน =====" | tee -a "$LOG"

for name in "$@"; do
  i=$((i+1))
  cfg="configs/stage2_${name}.yaml"

  if [ -f "outputs/runs/${name}/metrics.json" ]; then
    echo "[$(date '+%H:%M:%S')] ($i/$total) ข้าม $name (มีผลแล้ว)" | tee -a "$LOG"
    continue
  fi
  if [ ! -f "$cfg" ]; then
    echo "[$(date '+%H:%M:%S')] ($i/$total) FAIL $name — ไม่พบ $cfg" | tee -a "$LOG"
    continue
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
" 2>/dev/null)
    status="OK  "
  else
    res="— ดู outputs/comparison/log_${name}.txt"
    status="FAIL"
  fi

  # ประมาณเวลาที่เหลือจากค่าเฉลี่ยจริงของรันที่ผ่านมา ไม่ใช่ค่าที่เดาไว้ล่วงหน้า
  el=$(( ($(date +%s) - t0) / 60 ))
  eta=$(( el * (total - i) / i ))
  echo "[$(date '+%H:%M:%S')] ($i/$total) $status $name  ${dur}min  $res  | เหลืออีก ~${eta} นาที" | tee -a "$LOG"
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] ===== รันครบแล้ว กำลังทำรายงาน =====" | tee -a "$LOG"
$PY scripts/aggregate_seeds.py --prefix effb0_cmp_   --out outputs/comparison/SEEDS_cmp.md        > /dev/null 2>&1
$PY scripts/aggregate_seeds.py --prefix effb0_sw_pp_ --out outputs/comparison/SEEDS_preprocess.md > /dev/null 2>&1
$PY scripts/aggregate_seeds.py --prefix effb0_sw_aug_ --out outputs/comparison/SEEDS_augment.md   > /dev/null 2>&1
$PY scripts/make_summary_report.py > /dev/null 2>&1
$PY scripts/compare_runs.py        > /dev/null 2>&1
echo "[$(date '+%Y-%m-%d %H:%M:%S')] ===== เสร็จทั้งหมด =====" | tee -a "$LOG"
