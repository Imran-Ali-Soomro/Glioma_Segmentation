#!/bin/bash
cd "$(dirname "$0")"
export KMP_DUPLICATE_LIB_OK=TRUE
PY="${PYTHON:-python}"
REV_VARIANT=slide "$PY" h3_heldout_eval.py > logs/h3_slide2.log 2>&1
echo "h3 slide done $(date)" > logs/chain_heldout2.log
REV_VARIANT=tiled "$PY" h3_heldout_eval.py > logs/h3_tiled2.log 2>&1
echo "CHAIN_DONE $(date)" >> logs/chain_heldout2.log
