"""PTDR Stage 1.6, section 1: freeze/export full per-region cross-fitted
predictions from Stage 1.5's exact frozen model configs (Q_act: S2 MLP;
Q_transition: T0 S1 logistic regression, unweighted, source-conditioned
masking). This is a pure EXPORT step -- it reuses stage15_realistic_k5.py's
exact functions (same hyperparameters, same frozen `fold` column, same
random_state) to recover per-region probability vectors that were computed
in Stage 1.5 but only persisted as aggregate metrics, not full arrays.

No model selection, no new hyperparameters, no retraining of anything beyond
reproducing byte-for-byte the same cheap sklearn fits Stage 1.5 already ran.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1])); from mriseg_paths import P as _P, CODE as _CODE  # noqa: E402
REPO_ROOT = _P(r"")
sys.path.insert(0, str(REPO_ROOT / "analysis" / "ptdr_stage1"))

from stage15_realistic_k5 import (  # noqa: E402
    CLASS_NAMES, N_CLASSES, N_FOLDS,
    load_all_regions_with_oracle, learned_q_act_oof,
    learned_q_transition_oof_full_population, mask_and_renormalize,
)


def main():
    data = load_all_regions_with_oracle()
    n = data["n"]
    print(f"Loaded {n} predicted-FG regions (all, not just oracle-actionable)")

    # ---- reproduce Stage 1.5's exact best config: Q_act = S2 MLP ----
    print("Fitting learned Q_act (S2 MLP(16,), frozen-fold cross-fit, ALL regions)...")
    p_act = learned_q_act_oof(data["S2"], data["label_act"], data["fold"])

    # ---- reproduce Stage 1.5's exact best config: Q_transition = T0 (S1 logreg, unweighted) ----
    print("Fitting learned Q_transition (T0: S1 logreg, unweighted, frozen-fold, inference on ALL regions)...")
    oof_proba_raw, permitted_sets_per_fold = learned_q_transition_oof_full_population(
        data["S1"], data["label_act"], data["transition_target_oracle"], data["pred_class"], data["fold"],
        model_type="logreg", class_weight=None)
    masked_proba = mask_and_renormalize(oof_proba_raw, data["pred_class"], data["fold"], permitted_sets_per_fold)

    hard_gate_acted = p_act >= 0.5
    hard_gate_target = masked_proba.argmax(axis=1)

    # ---- sanity check against Stage 1.5's persisted realistic K5 numbers (T_act=0.5) ----
    cvc = data["class_voxel_counts"]
    pred_class_arr = data["pred_class"]
    idx_acted = np.flatnonzero(hard_gate_acted)
    chosen = hard_gate_target[idx_acted]
    orig = pred_class_arr[idx_acted]
    net_delta = np.array([cvc[i, chosen[j]] - cvc[i, orig[j]] for j, i in enumerate(idx_acted)])
    beneficial = net_delta > 0
    harmful = net_delta < 0
    print(f"\nSanity check (should match PTDR_STAGE15_REALISTIC_K5.json default T_act=0.5 point):")
    for cname, cidx in [("netc", 1), ("rc", 4)]:
        m = chosen == cidx
        print(f"  {cname}: n_acted={int(m.sum())} beneficial={int(beneficial[m].sum())} harmful={int(harmful[m].sum())}")

    with open(REPO_ROOT / "PTDR_STAGE15_REALISTIC_K5.json") as f:
        stage15_k5 = json.load(f)
    stage15_default = stage15_k5["default_operating_point_T_act_0.5"]
    sanity = {}
    for cname in ["netc", "rc"]:
        cidx = CLASS_NAMES.index(cname)
        m = chosen == cidx
        recomputed = {"n_acted": int(m.sum()), "beneficial": int(beneficial[m].sum()), "harmful": int(harmful[m].sum())}
        original = stage15_default["by_predicted_target"][cname]
        matches = (recomputed["n_acted"] == original["n_acted"] and recomputed["beneficial"] == original["beneficial"]
                   and recomputed["harmful"] == original["harmful"])
        sanity[cname] = {"recomputed": recomputed, "stage15_original": {k: original[k] for k in ["n_acted", "beneficial", "harmful"]},
                          "matches_exactly": matches}
        print(f"  {cname} matches Stage 1.5 exactly: {matches}")

    # ---- also export unmasked raw Q_transition proba (for calibration/diagnostics) ----
    df = pd.DataFrame({
        "patient": data["patients"], "case": [None] * n,  # case not tracked in load_all_regions_with_oracle
        "fold": data["fold"], "pred_class": pred_class_arr,
        "size_voxels_mm3": None,
        "label_act": data["label_act"], "transition_target_oracle": data["transition_target_oracle"],
        "p_act": p_act, "hard_gate_acted": hard_gate_acted, "hard_gate_target": hard_gate_target,
    })
    for i, cname in enumerate(CLASS_NAMES):
        df[f"class_voxel_count_{cname}"] = cvc[:, i]
        df[f"p_transition_masked_{cname}"] = masked_proba[:, i]
        df[f"p_transition_raw_{cname}"] = oof_proba_raw[:, i]

    # re-attach case/size from the raw action-labels file (load_all_regions_with_oracle doesn't carry them)
    with open(REPO_ROOT / "PTDR_3D_OOF_ACTION_LABELS.json") as f:
        oracle_full = json.load(f)
    assert len(oracle_full) == n
    df["case"] = [r["case"] for r in oracle_full]
    df["size_voxels_mm3"] = [r["size_voxels_mm3"] for r in oracle_full]
    df["region_id"] = np.arange(n)

    out_path = REPO_ROOT / "PTDR_STAGE16_PER_REGION_PREDICTIONS.parquet"
    df.to_parquet(out_path, index=False)
    print(f"\nWrote {out_path} ({len(df)} rows, {len(df.columns)} cols)")

    (REPO_ROOT / "PTDR_STAGE16_EXPORT_SANITY_CHECK.json").write_text(json.dumps(sanity, indent=2))
    print("Wrote PTDR_STAGE16_EXPORT_SANITY_CHECK.json")
    assert all(v["matches_exactly"] for v in sanity.values()), "Export does not reproduce Stage 1.5 exactly!"
    print("\nAll sanity checks PASSED -- exported predictions exactly reproduce Stage 1.5's realistic K5.")


if __name__ == "__main__":
    main()
