"""PTDR K6: shared data loading / label construction, reused by the training and
mechanical-verification scripts. Frozen population: 19,432 predicted-FG regions,
153 dev patients, 5 frozen patient-grouped folds (the `fold` column), frozen S2
representation, frozen oracle utility labels. No BG-recovery candidates.
"""
import json
from pathlib import Path

import numpy as np

import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1])); from mriseg_paths import P as _P, CODE as _CODE  # noqa: E402
REPO_ROOT = _P(r"")
CLASS_NAMES = ["bg", "netc", "snfh", "et", "rc"]
N_CLASSES = 5
N_FOLDS = 5


def load_k6_data():
    with open(REPO_ROOT / "PTDR_3D_OOF_ACTION_LABELS.json") as f:
        oracle = json.load(f)
    n = len(oracle)

    patients = np.array([r["patient"] for r in oracle])
    cases = np.array([r["case"] for r in oracle])
    fold = np.array([r["fold"] for r in oracle])
    pred_class = np.array([r["pred_class"] for r in oracle])  # source class c_r
    label_act = np.array([r["label_act"] for r in oracle])
    transition_target = np.array([r["transition_target"] if r["transition_target"] is not None else -1 for r in oracle])
    size_voxels = np.array([r["size_voxels_mm3"] for r in oracle], dtype=np.float64)
    class_voxel_counts = np.array([r["class_voxel_counts"] for r in oracle], dtype=np.float64)

    # y_final per PTDR_K6_FINAL_ACTION_LABEL_SPEC.md
    y_final = np.where(label_act == 0, pred_class, transition_target).astype(np.int64)

    # mechanical verification (governing task section 2 / 19's K6_PROTOCOL_FAILURE check)
    violation_mask = (y_final == pred_class) != (label_act == 0)
    n_violations = int(violation_mask.sum())

    # S2 feature construction (identical to Stage 1.5/1.6)
    S0 = np.array([r["mean_probs"] for r in oracle], dtype=np.float64)
    pred_onehot = np.zeros((n, 5))
    for i, r in enumerate(oracle):
        pred_onehot[i, r["pred_class"]] = 1.0
    log_size = np.log1p(size_voxels).reshape(-1, 1)
    S1 = np.concatenate([S0, pred_onehot, log_size], axis=1)
    decoder_feat_mean = np.array([r["decoder_feat_mean"] for r in oracle], dtype=np.float64)
    decoder_feat_std = np.array([r["decoder_feat_std"] for r in oracle], dtype=np.float64)
    S2 = np.concatenate([S1, decoder_feat_mean, decoder_feat_std], axis=1)

    return {
        "n": n, "patients": patients, "cases": cases, "fold": fold, "pred_class": pred_class,
        "label_act": label_act, "transition_target": transition_target, "y_final": y_final,
        "size_voxels": size_voxels, "class_voxel_counts": class_voxel_counts, "S2": S2,
        "n_label_violations": n_violations, "violation_mask": violation_mask,
    }


if __name__ == "__main__":
    d = load_k6_data()
    print(f"n={d['n']}, S2 dim={d['S2'].shape[1]}, n_label_violations={d['n_label_violations']}")
    verdict = "PASS" if d["n_label_violations"] == 0 else "K6_PROTOCOL_FAILURE"
    out = {"n_regions": d["n"], "n_violations": d["n_label_violations"], "verdict": verdict}
    (REPO_ROOT / "PTDR_K6_LABEL_VERIFICATION_RAW.json").write_text(json.dumps(out, indent=2))
    print(f"Verdict: {verdict}")
