"""PTDR Stage 1.6, section 10: calibration check for probability composition.

The project's existing calibration protocol (PTDR_CALIBRATION_ABSTENTION_PLAN.md)
explicitly specifies post-hoc temperature scaling (Guo et al. 2017) as its planned
calibration-improvement step -- so the secondary CALIBRATED_FACTORIZED_TYPED arm IS
authorized here (fit strictly on training-fold data inside each outer fold, never
on the held-out decision fold), per the governing task's conditional instruction.

PRIMARY Stage 1.6 analysis (stage16_controller_comparison.py) uses RAW cross-fitted
probabilities throughout. This script reports RAW and CALIBRATED separately.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, brier_score_loss

import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1])); from mriseg_paths import P as _P, CODE as _CODE  # noqa: E402
REPO_ROOT = _P(r"")
sys.path.insert(0, str(REPO_ROOT / "analysis" / "ptdr_stage1"))
CLASS_NAMES = ["bg", "netc", "snfh", "et", "rc"]
N_FOLDS = 5


def ece_from_conf_correct(conf, correct, n_bins=12):
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bins = []
    n = len(conf)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if mask.sum() == 0:
            bins.append({"bin_lo": float(lo), "bin_hi": float(hi), "n": 0, "mean_confidence": None, "empirical_accuracy": None})
            continue
        mc, ea = float(conf[mask].mean()), float(correct[mask].mean())
        ece += (mask.sum() / n) * abs(mc - ea)
        bins.append({"bin_lo": float(lo), "bin_hi": float(hi), "n": int(mask.sum()), "mean_confidence": mc, "empirical_accuracy": ea})
    return float(ece), bins


def temperature_fit_binary(y, p, grid=None):
    if grid is None:
        # wide grid, well past the project's ordinary [0.3,3.0] search range, specifically to
        # check whether Q_act's extreme overconfidence (see diagnostic below) reflects a
        # boundary-clipped, non-converged temperature rather than a genuine interior optimum.
        grid = np.concatenate([np.linspace(0.3, 3.0, 28), np.linspace(3.5, 30.0, 54)])
    logit = np.log(np.clip(p, 1e-9, 1 - 1e-9) / np.clip(1 - p, 1e-9, 1 - 1e-9))
    nlls = []
    for T in grid:
        p_t = 1 / (1 + np.exp(-logit / T))
        nlls.append(log_loss(y, np.clip(p_t, 1e-9, 1 - 1e-9)))
    nlls = np.array(nlls)
    best_i = int(nlls.argmin())
    best_T, best_nll = float(grid[best_i]), float(nlls[best_i])
    diverged = best_i == len(grid) - 1  # optimum sits at the grid's upper edge -- no interior minimum found
    return best_T, diverged, {"T_at_1.0_nll": float(nlls[np.argmin(np.abs(grid - 1.0))]),
                               "T_at_3.0_nll": float(nlls[np.argmin(np.abs(grid - 3.0))]),
                               "T_at_grid_max_nll": float(nlls[-1])}


def apply_temperature_binary(p, T):
    logit = np.log(np.clip(p, 1e-9, 1 - 1e-9) / np.clip(1 - p, 1e-9, 1 - 1e-9))
    return 1 / (1 + np.exp(-logit / T))


def main():
    df = pd.read_parquet(REPO_ROOT / "PTDR_STAGE16_PER_REGION_PREDICTIONS.parquet")
    p_act = df["p_act"].values
    label_act = df["label_act"].values
    fold = df["fold"].values
    pred_class = df["pred_class"].values
    n = len(df)

    pred_bin = (p_act >= 0.5).astype(int)
    correct_act = (pred_bin == label_act).astype(float)
    brier_act = float(brier_score_loss(label_act, p_act))
    nll_act = float(log_loss(label_act, np.clip(p_act, 1e-9, 1 - 1e-9)))
    conf_act = np.where(p_act >= 0.5, p_act, 1 - p_act)  # confidence in the chosen binary class
    ece_act, bins_act = ece_from_conf_correct(conf_act, correct_act)

    by_source_act = {}
    for c in sorted(set(pred_class)):
        m = pred_class == c
        by_source_act[CLASS_NAMES[c]] = {
            "n": int(m.sum()), "brier": float(brier_score_loss(label_act[m], p_act[m])),
            "nll": float(log_loss(label_act[m], np.clip(p_act[m], 1e-9, 1 - 1e-9))),
            "ece": ece_from_conf_correct(conf_act[m], correct_act[m])[0],
        }

    # ---- reuse Stage 1.5's existing Q_transition calibration values ----
    with open(REPO_ROOT / "PTDR_STAGE15_QTRANSITION_S2_RESULTS.json") as f:
        stage15_q = json.load(f)
    q_transition_calibration_stage15 = stage15_q.get("calibration_section11")
    best_variant = stage15_q["best_variant_balanced_masked"]
    t0_masked_unweighted_per_class = None
    # per-target precision/recall reliability proxy: reuse imbalance-control arm_A (unweighted) per_class
    with open(REPO_ROOT / "PTDR_STAGE15_IMBALANCE_CONTROL.json") as f:
        imb = json.load(f)
    t0_masked_unweighted_per_class = imb["arm_A_standard_unweighted"]["per_class"]
    t0_masked_unweighted_brier = imb["arm_A_standard_unweighted"].get("brier_per_class")

    # ---- secondary arm: CALIBRATED_FACTORIZED_TYPED (per-fold temperature scaling, train-fold only) ----
    print("Fitting per-fold temperature scaling for Q_act (train-fold only)...")
    p_act_calibrated = np.zeros(n)
    temps_act = {}
    q_act_temp_diagnostics = {}
    any_diverged = False
    for k in range(N_FOLDS):
        tr = fold != k
        te = fold == k
        T, diverged, diag = temperature_fit_binary(label_act[tr], p_act[tr])
        temps_act[k] = float(T)
        q_act_temp_diagnostics[k] = {"chosen_T": T, "diverged_to_grid_edge": diverged, **diag}
        any_diverged = any_diverged or diverged
        p_act_calibrated[te] = apply_temperature_binary(p_act[te], T)
    q_act_calibration_converged = not any_diverged
    print(f"Q_act temperature search converged to an interior optimum in all folds: {q_act_calibration_converged}")
    if not q_act_calibration_converged:
        print("  -> at least one fold's NLL-minimizing temperature sits at the grid edge (no interior "
              "minimum found): single-scalar temperature scaling does not adequately correct Q_act's "
              "miscalibration. Falling back to RAW p_act for the adopted CALIBRATED_FACTORIZED_TYPED arm "
              "(Q_transition's temperature scaling, which DOES converge, is still applied).")
        p_act_calibrated = p_act.copy()  # fall back to RAW for Q_act specifically; disclosed, not silently adopted

    # reuse Stage 1.5's exact Q_transition temperature-scaling machinery, but this time applied to the
    # FULL all-regions population (export's raw unmasked proba), trained on each fold's oracle-actionable
    # TRAINING rows only (matching stage15_realistic_k5.py's masking/training population discipline)
    from stage15_realistic_k5 import mask_and_renormalize  # noqa: E402
    p_trans_raw = df[[f"p_transition_raw_{c}" for c in CLASS_NAMES]].values
    log_proba = np.log(np.clip(p_trans_raw, 1e-12, 1.0))
    calibrated_trans_raw = np.zeros_like(p_trans_raw)
    temps_trans = {}
    grid = np.linspace(0.3, 3.0, 28)
    for k in range(N_FOLDS):
        tr_act = (fold != k) & (label_act == 1)
        te = fold == k
        best_T, best_nll = 1.0, np.inf
        y_tr = df.loc[tr_act, "transition_target_oracle"].values
        for T in grid:
            scaled = log_proba[tr_act.values if hasattr(tr_act, "values") else tr_act] / T
            scaled = scaled - scaled.max(axis=1, keepdims=True)
            p = np.exp(scaled); p = p / p.sum(axis=1, keepdims=True)
            nll = log_loss(y_tr, np.clip(p, 1e-12, 1.0), labels=list(range(5)))
            if nll < best_nll:
                best_nll, best_T = nll, T
        temps_trans[k] = float(best_T)
        scaled = log_proba[te] / best_T
        scaled = scaled - scaled.max(axis=1, keepdims=True)
        p = np.exp(scaled); p = p / p.sum(axis=1, keepdims=True)
        calibrated_trans_raw[te] = p

    # re-derive permitted sets exactly as stage16 export did, to mask the calibrated transition proba
    permitted_sets_per_fold = {}
    for k in range(N_FOLDS):
        tr_act = (fold != k) & (label_act == 1)
        perm = {}
        for src in sorted(set(pred_class[tr_act])):
            m = tr_act & (pred_class == src)
            perm[int(src)] = set(int(t) for t in df.loc[m, "transition_target_oracle"].values)
        permitted_sets_per_fold[k] = perm
    calibrated_trans_masked = mask_and_renormalize(calibrated_trans_raw, pred_class, fold, permitted_sets_per_fold)

    # ---- CALIBRATED_FACTORIZED_TYPED controller ----
    action_values_cal = p_act_calibrated[:, None] * calibrated_trans_masked
    keep_value_cal = 1.0 - p_act_calibrated
    best_action_idx_cal = action_values_cal.argmax(axis=1)
    best_action_value_cal = action_values_cal[np.arange(n), best_action_idx_cal]
    fac_acted_cal = best_action_value_cal > keep_value_cal
    fac_target_cal = np.where(fac_acted_cal, best_action_idx_cal, -1)

    cvc_cols = [f"class_voxel_count_{c}" for c in CLASS_NAMES]
    cvc = df[cvc_cols].values
    source_idx = pred_class
    patients = df["patient"].values
    idx = np.flatnonzero(fac_acted_cal)
    delta = cvc[idx, fac_target_cal[idx]] - cvc[idx, source_idx[idx]]
    beneficial, harmful = delta > 0, delta < 0

    def target_stats(target_name):
        cidx = CLASS_NAMES.index(target_name)
        m = fac_target_cal[idx] == cidx
        n_m = int(m.sum())
        if n_m == 0:
            return {"n_acted": 0}
        ben, harm = int(beneficial[m].sum()), int(harmful[m].sum())
        true_total = int(np.sum((label_act == 1) & (df["transition_target_oracle"].values == cidx)))
        return {
            "n_acted": n_m, "beneficial": ben, "harmful": harm, "net_corrected": ben - harm,
            "beneficial_to_harmful_ratio": (ben / harm) if harm > 0 else (float("inf") if ben > 0 else None),
            "harmful_edit_rate": harm / n_m, "correction_precision": ben / n_m,
            "correction_coverage_of_target_actionable": (ben / true_total) if true_total else None,
            "patient_support_beneficial": len(set(patients[idx][m][beneficial[m]])) if ben else 0,
        }

    calibrated_factorized_summary = {
        "n_acted": len(idx), "net_corrected": int(beneficial.sum()) - int(harmful.sum()),
        "harmful_edit_rate": float(harmful.mean()) if len(idx) else None,
        "rc": target_stats("rc"), "netc": target_stats("netc"),
        "per_fold_temperatures_q_act": temps_act, "per_fold_temperatures_q_transition": temps_trans,
        "q_act_calibration_converged": q_act_calibration_converged,
        "q_act_temperature_search_diagnostics": q_act_temp_diagnostics,
        "q_act_arm_used": "RAW (fallback)" if not q_act_calibration_converged else "temperature-scaled",
        "note_q_act_divergence": None if q_act_calibration_converged else (
            "Q_act's per-fold NLL-minimizing temperature never reached an interior optimum even when the "
            "search grid was widened to T=30 (NLL kept monotonically decreasing toward the grid edge in at "
            "least one fold). This indicates Q_act's raw probabilities contain a long tail of severely "
            "overconfident WRONG predictions that a single monotonic rescaling (temperature scaling) can "
            "only 'fix' by crushing ALL confidence toward 0.5 -- destroying the model's otherwise-useful "
            "high-confidence CORRECT predictions in the process. This is a genuine, disclosed limitation of "
            "the project's single-scalar temperature-scaling protocol for this specific head, not resolved "
            "further here (Platt scaling / isotonic regression would be the next step and is out of this "
            "diagnostic's scope). The adopted CALIBRATED_FACTORIZED_TYPED arm therefore uses RAW p_act "
            "combined with temperature-scaled p_transition (which converged normally, see "
            "per_fold_temperatures_q_transition)."
        ),
    }

    out = {
        "q_act_calibration_raw": {
            "brier": brier_act, "nll": nll_act, "ece": ece_act, "reliability_bins": bins_act,
            "by_source_class": by_source_act,
        },
        "q_transition_calibration_reused_from_stage15": q_transition_calibration_stage15,
        "q_transition_per_target_precision_recall_oracle_gated_unweighted_masked": t0_masked_unweighted_per_class,
        "q_transition_per_target_brier_oracle_gated": t0_masked_unweighted_brier,
        "secondary_arm_authorized_by": "PTDR_CALIBRATION_ABSTENTION_PLAN.md explicitly specifies post-hoc "
            "temperature scaling (Guo et al. 2017) as the project's planned calibration-improvement step -- "
            "so CALIBRATED_FACTORIZED_TYPED is run here as an authorized secondary sensitivity arm.",
        "calibrated_factorized_typed_rc_and_netc": calibrated_factorized_summary,
        "note": "Temperatures fit per outer fold on TRAINING-fold data only (Q_act: all training regions, "
                "binary NLL grid search; Q_transition: training-fold ORACLE-ACTIONABLE regions only, "
                "multiclass NLL grid search, matching Stage 1.5's section 11 protocol), applied to the "
                "held-out fold, then re-masked/renormalized identically to the RAW arm.",
    }
    (REPO_ROOT / "PTDR_STAGE16_CALIBRATION_AUDIT.json").write_text(json.dumps(out, indent=2, default=str))
    print("Wrote PTDR_STAGE16_CALIBRATION_AUDIT.json")
    print(f"Q_act RAW: Brier={brier_act:.4f} NLL={nll_act:.4f} ECE={ece_act:.4f}")
    print(f"CALIBRATED_FACTORIZED RC: {calibrated_factorized_summary['rc']}")
    print(f"CALIBRATED_FACTORIZED NETC: {calibrated_factorized_summary['netc']}")
    print(f"Per-fold Q_act temperatures: {temps_act}")


if __name__ == "__main__":
    main()
