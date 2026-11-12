"""PTDR Stage 1: minimal custom trainer for OOF base-segmenter generation.

Architecture is 100% stock nnU-Net ResidualEncoderUNet (inherited unchanged from
nnUNetTrainer_100epochs -- no gating, no attention, no PTDR modules, per the
explicit protocol constraint). The ONLY override is do_split(), which reads
from PTDR_3D_OOF_SPLITS_DERIVED.json (5 patient-grouped folds built specifically
for this OOF exercise, NOT the same as the canonical splits_final.json) instead
of the stock split file -- same pattern used for the earlier minimal 3D B0/B1
confirmation's do_split() override, and for the same reason (reuse the
existing clean split machinery without touching the canonical artifact).

Launch identically to normal nnU-Net cross-validation:
    nnUNetv2_train 1 3d_fullres 0 -tr nnUNetTrainerPTDROOF_100epochs -p nnUNetResEncUNetMPlans
    nnUNetv2_train 1 3d_fullres 1 -tr nnUNetTrainerPTDROOF_100epochs -p nnUNetResEncUNetMPlans
    ... (fold 2, 3, 4)
self.fold (set by nnU-Net from the -f/positional fold CLI arg) selects which
of the 5 derived folds' val-patients this run generates OOF predictions for.
"""
import json
import os
from pathlib import Path

from nnunetv2.training.nnUNetTrainer.variants.training_length.nnUNetTrainer_Xepochs import nnUNetTrainer_100epochs

DERIVED_SPLITS_PATH = Path(os.environ["MRISEG_ROOT"]) / "nnUNet_data" / "nnUNet_preprocessed" / "Dataset001_MUGliomaPost" / "PTDR_3D_OOF_SPLITS_DERIVED.json"


class nnUNetTrainerPTDROOF_100epochs(nnUNetTrainer_100epochs):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, device=None):
        import torch
        super().__init__(plans, configuration, fold, dataset_json,
                          device if device is not None else torch.device("cuda"))
        # Frequent checkpointing (every 2 epochs, vs nnU-Net's default 50): this environment
        # has been observed to kill long-running background training tasks unpredictably
        # (session-restart-independent -- a single background run was terminated after only
        # ~13-15 minutes with no apparent duration-related cause). Checkpointing every 2
        # epochs (~4-5 minutes of wall-clock) bounds the maximum lost work per interruption
        # to a small, resumable amount via --c, instead of needing a full restart each time.
        self.save_every = 2

    def do_split(self):
        """Overrides the stock do_split to read from PTDR_3D_OOF_SPLITS_DERIVED.json
        (5 NEW patient-grouped OOF folds, built for PTDR Stage 1 specifically -- distinct
        from the canonical splits_final.json, which is NOT modified by this trainer)."""
        with open(DERIVED_SPLITS_PATH) as f:
            splits = json.load(f)
        tr_keys = splits[self.fold]["train"]
        val_keys = splits[self.fold]["val"]
        self.print_to_log_file(
            f"PTDR STAGE 1 OOF SPLIT (not canonical splits_final.json): {DERIVED_SPLITS_PATH}. "
            f"Fold {self.fold}: {len(tr_keys)} train / {len(val_keys)} val cases."
        )
        assert not (set(tr_keys) & set(val_keys)), "train/val overlap in derived OOF split -- must not happen"
        return tr_keys, val_keys


class nnUNetTrainerPTDROOF_5epochs(nnUNetTrainerPTDROOF_100epochs):
    """Technical preflight only (timing/VRAM/AMP stability) -- section 9. Never used for a
    real OOF fold's final predictions."""

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, device=None):
        import torch
        super().__init__(plans, configuration, fold, dataset_json,
                          device if device is not None else torch.device("cuda"))
        self.num_epochs = 5
