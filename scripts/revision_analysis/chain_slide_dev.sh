#!/bin/bash
cd "$(dirname "$0")"
export KMP_DUPLICATE_LIB_OK=TRUE REV_VARIANT=slide
PY="${PYTHON:-python}"
until [ -f logs/extract_slide_dev_DONE.txt ]; do sleep 10; done
echo "extraction done $(date)" > logs/chain_slide.log
powershell -NoProfile -ExecutionPolicy Bypass -File run_a2_parallel.ps1
echo "a2 done $(date)" >> logs/chain_slide.log
"$PY" a3_compare_models.py > logs/slide_a3.log 2>&1
echo "a3 done $(date)" >> logs/chain_slide.log
"$PY" a1_protocol_audit.py > logs/slide_a1.log 2>&1 &
"$PY" a4_calibration_rc.py > logs/slide_a4.log 2>&1 &
"$PY" a5_ceiling_decomposition.py > logs/slide_a5.log 2>&1 &
"$PY" a6_dice_hd95.py --set dev --workers 8 > logs/slide_a6.log 2>&1
"$PY" a7_dice_summary_dev.py > logs/slide_a7.log 2>&1
wait
echo "CHAIN_DONE $(date)" >> logs/chain_slide.log
