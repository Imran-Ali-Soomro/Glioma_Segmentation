#!/bin/bash
cd "$(dirname "$0")"
export KMP_DUPLICATE_LIB_OK=TRUE REV_VARIANT=slide
PY="${PYTHON:-python}"
"$PY" a6_dice_hd95.py --set dev --workers 8 > logs/slide_a6.log 2>&1
"$PY" a7_dice_summary_dev.py > logs/slide_a7.log 2>&1
"$PY" a3_compare_models.py > logs/slide_a3b.log 2>&1
echo "DICE_CHAIN_DONE $(date)" > logs/chain_dice_slide.log
