#!/usr/bin/env bash
# run_batch.sh — รันชุด config ต่อกันไป พร้อมบันทึกความคืบหน้าลงไฟล์
#
# ทำไมต้องมี: การกวาดพารามิเตอร์รอบนี้ใช้เวลาข้ามคืน ถ้ารันเป็นก้อนเดียวแล้วเงียบ
# ไปทั้งคืน จะไม่รู้เลยว่าถึงไหนแล้ว หรือมีรันไหนล้มไปตั้งแต่ชั่วโมงแรก
# ไฟล์นี้เขียนบรรทัดสรุปต่อ 1 รันลง PROGRESS.log ทันทีที่รันนั้นจบ เปิดดูได้ตลอด
#
# ออกแบบให้ "ล้มแล้วไปต่อ" — รันไหนพังจะบันทึกว่า FAIL แล้วขึ้นรันถัดไป
# ไม่หยุดทั้งสาย เพราะเสียเวลาทั้งคืนเพราะรันเดียวล้มไม่คุ้ม
#
# USAGE: bash scripts/run_batch.sh <ชื่อ batch> <config1> <config2> ...

set -u
BATCH="$1"; shift
PY=./.venv/Scripts/python.exe
LOG=outputs/comparison/PROGRESS.log
mkdir -p outputs/comparison
export PYTHONIOENCODING=utf-8

# --- กันสองก้อนรันชนกัน ---
# มี GPU ตัวเดียว ถ้า batch สองก้อนเริ่มพร้อมกันจะแย่ง VRAM กันจนช้าทั้งคู่ หรือ
# หนักกว่านั้นคือรันชื่อเดียวกันพร้อมกันแล้วเขียนทับผลกันเอง
# ใช้ mkdir เป็น mutex เพราะมันเป็น atomic operation ในระบบไฟล์ (ต่างจากการเช็ค
# -f แล้วค่อย touch ซึ่งมีช่องว่างให้สองโปรเซสผ่านพร้อมกันได้)
LOCK=outputs/comparison/.batch.lock
until mkdir "$LOCK" 2>/dev/null; do
  echo "[$(date '+%H:%M:%S')] batch '$BATCH' รอคิว (มี batch อื่นรันอยู่)..." >> "$LOG"
  sleep 60
done
trap 'rmdir "$LOCK" 2>/dev/null' EXIT INT TERM

total=$#
i=0
echo "[$(date '+%Y-%m-%d %H:%M:%S')] === เริ่ม batch '$BATCH' ($total รัน) ===" | tee -a "$LOG"

for name in "$@"; do
  i=$((i+1))
  cfg="configs/stage2_${name}.yaml"
  start=$(date +%s)

  # ข้ามรันที่ทำไปแล้ว เผื่อต้องรันสคริปต์ซ้ำหลังเครื่องดับกลางคัน
  if [ -f "outputs/runs/${name}/metrics.json" ]; then
    echo "[$(date '+%H:%M:%S')] ($i/$total) ข้าม $name (มีผลแล้ว)" | tee -a "$LOG"
    continue
  fi

  if [ ! -f "$cfg" ]; then
    echo "[$(date '+%H:%M:%S')] ($i/$total) FAIL $name — ไม่พบ $cfg" | tee -a "$LOG"
    continue
  fi

  $PY scripts/train.py --config "$cfg" > "outputs/comparison/log_${name}.txt" 2>&1
  dur=$(( ($(date +%s) - start) / 60 ))

  if [ -f "outputs/runs/${name}/metrics.json" ]; then
    line=$($PY -c "
import json
m=json.load(open('outputs/runs/${name}/metrics.json',encoding='utf-8'))
r=m['per_grade_recall']; t=m['test_metrics']
print(f\"AUC={t['auc']:.3f} mild={r['mild']['recall']:.3f} mod={r['moderate']['recall']:.3f} F1={t['macro_f1']:.3f} ep={m['best_epoch']}\")
" 2>/dev/null)
    echo "[$(date '+%H:%M:%S')] ($i/$total) OK   $name  ${dur}min  $line" | tee -a "$LOG"
  else
    echo "[$(date '+%H:%M:%S')] ($i/$total) FAIL $name  ${dur}min — ดู outputs/comparison/log_${name}.txt" | tee -a "$LOG"
  fi
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] === จบ batch '$BATCH' ===" | tee -a "$LOG"
