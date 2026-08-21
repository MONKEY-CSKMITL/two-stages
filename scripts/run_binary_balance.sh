#!/usr/bin/env bash
# run_binary_balance.sh — โจทย์ binary: เทียบวิธีทำให้ข้อมูลสมดุล 4 แบบ
#
# ที่มา: วัดแล้วพบว่าทุกวิธีที่ "ขยายข้อมูลคลาสน้อย" ทำให้จับคลาสน้อยได้แย่ลง
#   bin_orig      น้ำหนักคลาสอย่างเดียว        mild 0.511  (ตัวคุม)
#   bin_x10       อบ augment ลงไฟล์ + น้ำหนัก   mild 0.370
#   bin_rep10     ทำซ้ำแถว + น้ำหนัก            mild 0.294
#   bin_x10w      + ดันน้ำหนักอีก               mild 0.294
# สมมติฐาน: สูตร w = N/(K*n_c) ทำให้การทำซ้ำไปลดน้ำหนักต่อตัวอย่างของคลาสน้อย
# จาก 5.96 เหลือ 1.05 (5.7 เท่า) เคสยากๆ จึงถูกเจือจางจนโมเดลไม่ถูกบังคับให้แก้
#
# 6 รันนี้ทดสอบสมมติฐานนั้นโดยตัดตัวแปรน้ำหนักออกไปเลย:
#   bin_rep10_nw   ทำซ้ำ + ปิดน้ำหนัก      -> สมดุล 1.09:1 จากจำนวนแถวจริง
#   bin_ds11       down normal + ปิดน้ำหนัก -> 1:1 เป๊ะ  705 : 705 ต่อ epoch
#   bin_rep10_ds11 ทำซ้ำ + down + ปิดน้ำหนัก -> 1:1 เป๊ะ 7,050 : 7,050 ต่อ epoch
# ทั้งหมดทุกตัวอย่างน้ำหนัก 1.0 เท่ากัน ไม่มีการเจือจาง
#
# ล้มแล้วไปต่อ · รันซ้ำได้ (ข้ามรันที่มี metrics.json แล้ว)
# USAGE: bash scripts/run_binary_balance.sh

set -u
cd "$(dirname "$0")/.."
PY=./.venv/Scripts/python.exe
LOG=outputs/comparison/PROGRESS.log
export PYTHONIOENCODING=utf-8

echo "" | tee -a "$LOG"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] ===== binary: เทียบวิธีทำให้สมดุล =====" | tee -a "$LOG"

# สลับ tag ทีละ seed เพื่อให้ได้คู่ที่เทียบกันได้เร็วที่สุด หยุดกลางทางก็ยังสมดุล
for s in 42 43; do
for tag in rep10_nw ds11 rep10_ds11; do
  name="effb0_bin_${tag}_s${s}"
  if [ -f "outputs/runs/${name}/metrics.json" ]; then
    echo "[$(date '+%H:%M:%S')] ข้าม $name (มีผลแล้ว)" | tee -a "$LOG"; continue
  fi
  start=$(date +%s)
  $PY scripts/train.py --config "configs/stage2_${name}.yaml" > "outputs/comparison/log_${name}.txt" 2>&1
  dur=$(( ($(date +%s) - start) / 60 ))
  if [ -f "outputs/runs/${name}/metrics.json" ]; then
    res=$($PY -c "
import json
m=json.load(open('outputs/runs/${name}/metrics.json',encoding='utf-8'))
g=m['per_grade_recall']; t=m['test_metrics']; cm=m['confusion_matrix']
print(f\"F1={t['macro_f1']:.3f} AUC={t['auc']:.3f} mild={g['mild']['recall']:.3f} mod={g['moderate']['recall']:.3f} norm={g['normal']['recall']:.3f} หลุด={cm[1][0]} FP={cm[0][1]} ep={m['best_epoch']}\")" 2>/dev/null)
    echo "[$(date '+%H:%M:%S')] OK   $name  ${dur}min  $res" | tee -a "$LOG"
  else
    echo "[$(date '+%H:%M:%S')] FAIL $name  ${dur}min — ดู outputs/comparison/log_${name}.txt" | tee -a "$LOG"
  fi
  $PY scripts/aggregate_seeds.py --prefix effb0_bin_ --out outputs/comparison/SEEDS_binary.md >/dev/null 2>&1
done
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] ===== binary สมดุล เสร็จแล้ว =====" | tee -a "$LOG"
$PY scripts/make_summary_report.py >/dev/null 2>&1
