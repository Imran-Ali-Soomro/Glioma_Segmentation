#!/bin/bash
cd "$(dirname "$0")"
export KMP_DUPLICATE_LIB_OK=TRUE
PY="${PYTHON:-python}"
until [ -f logs/extract_heldout_DONE.txt ]; do sleep 15; done
powershell -NoProfile -ExecutionPolicy Bypass -File run_extract_slide.ps1 -Set heldout
echo "heldout extraction (tiled+slide) done $(date)" > logs/chain_heldout.log
until grep -q CHAIN_DONE logs/chain_slide.log 2>/dev/null; do sleep 15; done
REV_VARIANT=slide "$PY" h3_heldout_eval.py > logs/h3_slide.log 2>&1
echo "h3 slide done $(date)" >> logs/chain_heldout.log
REV_VARIANT=tiled "$PY" h3_heldout_eval.py > logs/h3_tiled.log 2>&1
echo "CHAIN_DONE $(date)" >> logs/chain_heldout.log
