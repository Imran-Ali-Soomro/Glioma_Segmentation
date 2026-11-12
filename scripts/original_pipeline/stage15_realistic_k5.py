"""PTDR Stage 1.5, sections 12-13: realistic K5 -- remove the oracle-target
assumption entirely. Learned Q_act (reproduced S2 MLP, matching Stage-1 K2's best
feature set/model, but cross-fit on the FROZEN `fold` column for consistency with
the rest of Stage 1.5) + learned, source-conditioned Q_transition (the best masked
T-variant from stage15_qtransition.py, refit identically but with inference
extended to ALL predicted-FG regions, not just the oracle-actionable subset) +
the existing Mechanism-A deterministic controller logic from
k5_and_bg_recovery_and_geometry.py (threshold-gated act/no-act, direct relabel),
with the LEARNED Q_transition's top prediction substituted at the "would this
edit be applied" decision point instead of the oracle's true target.

No oracle target is used at any inference-time decision in this script. Ground
truth (class_voxel_counts) is used ONLY to evaluate whether an already-applied
edit was beneficial/harmful after the fact -- exactly as in the original K5 and
Stage-1 K1/K2's own methodology (evaluation-only use of GT, never a decision input).
"""
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1])); from mriseg_paths import P as _P, CODE as _CODE  # noqa: E402
REPO_ROOT = _P(r"")
CLASS_NAMES = ["bg", "netc", "snfh", "et", "rc"]
N_CLASSES = 5
N_FOLDS = 5
ACT_THRESHOLD_GRID = [0.3, 0.4, 0.5, 0.6, 0.7]


def load_all_regions_with_oracle():
    with open(REPO_ROOT / "PTDR_3D_OOF_ACTION_LABELS.json") as f:
        oracle = json.load(f)
    patients = np.array([r["patient"] for r in oracle])
    fold = np.array([r["fold"] for r in oracle])
    pred_class = np.array([r["pred_class"] for r in oracle])
    label_act = np.array([r["label_act"] for r in oracle])
    transition_target_oracle = np.array([r["transition_target"] if r["transition_target"] is not None else -1 for r in oracle])
    class_voxel_counts = np.array([r["class_voxel_counts"] for r in oracle], dtype=np.float64)

    S0 = np.array([r["mean_probs"] for r in oracle], dtype=np.float64)
    pred_onehot = np.zeros((len(oracle), 5))
    for i, r in enumerate(oracle):
        pred_onehot[i, r["pred_class"]] = 1.0
    log_size = np.log1p(np.array([r["size_voxels_mm3"] for r in oracle], dtype=np.float64)).reshape(-1, 1)
    S1 = np.concatenate([S0, pred_onehot, log_size], axis=1)
    decoder_feat_mean = np.array([r["decoder_feat_mean"] for r in oracle], dtype=np.float64)
    decoder_feat_std = np.array([r["decoder_feat_std"] for r in oracle], dtype=np.float64)
    S2 = np.concatenate([S1, decoder_feat_mean, decoder_feat_std], axis=1)

    return {
        "patients": patients, "fold": fold, "pred_class": pred_class, "label_act": label_act,
        "transition_target_oracle": transition_target_oracle, "class_voxel_counts": class_voxel_counts,
        "S1": S1, "S2": S2, "n": len(oracle),
    }


def align_proba(proba, classes_, n_classes=N_CLASSES):
    out = np.zeros((proba.shape[0], n_classes))
    for j, c in enumerate(classes_):
        out[:, int(c)] = proba[:, j]
    return out


def learned_q_act_oof(S2, label_act, fold_arr):
    """Reproduce Stage-1 K2's best model (S2 MLP), cross-fit on the FROZEN fold
    column (sections 6-13's consistent CV discipline), over ALL regions."""
    n = len(label_act)
    oof = np.zeros(n)
    for k in range(N_FOLDS):
        te = fold_arr == k
        tr = ~te
        scaler = StandardScaler().fit(S2[tr])
        mlp = MLPClassifier(hidden_layer_sizes=(16,), max_iter=500, random_state=0,
                             early_stopping=True).fit(scaler.transform(S2[tr]), label_act[tr])
        proba = mlp.predict_proba(scaler.transform(S2[te]))
        col1 = list(mlp.classes_).index(1) if 1 in mlp.classes_ else None
        oof[te] = proba[:, col1] if col1 is not None else 0.0
    return oof


def learned_q_transition_oof_full_population(X, label_act, transition_target_oracle, pred_class_arr, fold_arr,
                                               model_type, class_weight):
    """Fit the best T-variant on each fold's TRAIN oracle-actionable rows only (the only
    rows with a defined supervision target), but apply predict_proba to ALL rows of the
    held-out fold (not just its oracle-actionable subset) -- this is what makes Q_transition
    usable as a real inference-time component instead of an oracle-gated one."""
    n = len(label_act)
    oof_proba = np.zeros((n, N_CLASSES))
    permitted_sets_per_fold = {}
    for k in range(N_FOLDS):
        te = fold_arr == k
        tr = ~te
        tr_act = tr & (label_act == 1)
        Xtr = X[tr_act]
        ytr = transition_target_oracle[tr_act]
        scaler = StandardScaler().fit(Xtr)
        Xtr_s = scaler.transform(Xtr)
        Xte_s = scaler.transform(X[te])
        if model_type == "logreg":
            clf = LogisticRegression(max_iter=2000, class_weight=class_weight).fit(Xtr_s, ytr)
        elif model_type == "mlp":
            clf = MLPClassifier(hidden_layer_sizes=(16, 16), max_iter=500, random_state=0,
                                 early_stopping=True).fit(Xtr_s, ytr)
        else:
            raise ValueError(model_type)
        proba = clf.predict_proba(Xte_s)
        oof_proba[te] = align_proba(proba, clf.classes_)

        permitted = {}
        for src in sorted(set(pred_class_arr[tr_act])):
            m = tr_act & (pred_class_arr == src)
            permitted[int(src)] = set(int(t) for t in transition_target_oracle[m])
        permitted_sets_per_fold[k] = permitted
    return oof_proba, permitted_sets_per_fold


def mask_and_renormalize(oof_proba, pred_class_arr, fold_arr, permitted_sets_per_fold):
    masked = np.zeros_like(oof_proba)
    for k in range(N_FOLDS):
        te = fold_arr == k
        perm = permitted_sets_per_fold[k]
        for src in sorted(set(pred_class_arr[te])):
            idx = np.flatnonzero(te & (pred_class_arr == src))
            allowed = sorted(perm.get(src, set(range(N_CLASSES))))
            if not allowed:
                allowed = list(range(N_CLASSES))
            sub = oof_proba[idx][:, allowed]
            rowsum = sub.sum(axis=1, keepdims=True)
            rowsum[rowsum == 0] = 1.0
            renorm = sub / rowsum
            full = np.zeros((len(idx), N_CLASSES))
            full[:, allowed] = renorm
            masked[idx] = full
    return masked


def evaluate_controller(q_act_prob, chosen_target, data, act_threshold, conf_threshold=None, top_conf=None):
    pred_class_arr = data["pred_class"]
    label_act = data["label_act"]
    cvc = data["class_voxel_counts"]
    patients = data["patients"]
    n = data["n"]

    acted = q_act_prob >= act_threshold
    if conf_threshold is not None:
        acted = acted & (top_conf >= conf_threshold)
    n_acted = int(acted.sum())
    if n_acted == 0:
        return {"act_threshold": act_threshold, "conf_threshold": conf_threshold, "n_acted": 0}

    idx = np.flatnonzero(acted)
    chosen = chosen_target[idx]
    orig = pred_class_arr[idx]
    net_delta = np.array([cvc[i, chosen[j]] - cvc[i, orig[j]] for j, i in enumerate(idx)])
    beneficial = net_delta > 0
    harmful = net_delta < 0
    n_beneficial = int(beneficial.sum())
    n_harmful = int(harmful.sum())
    n_unchanged_zero_delta = int((net_delta == 0).sum())

    n_unchanged_errors = int(((label_act == 1) & ~acted).sum())  # missed oracle-actionable opportunities

    overall = {
        "act_threshold": act_threshold, "conf_threshold": conf_threshold,
        "n_acted": n_acted, "beneficial_edits": n_beneficial, "harmful_edits": n_harmful,
        "zero_delta_edits": n_unchanged_zero_delta, "unchanged_errors_missed_opportunities": n_unchanged_errors,
        "net_corrected": n_beneficial - n_harmful,
        "beneficial_to_harmful_ratio": (n_beneficial / n_harmful) if n_harmful > 0 else (float("inf") if n_beneficial > 0 else None),
        "harmful_edit_rate": n_harmful / n_acted,
        "correction_precision": n_beneficial / n_acted,
        "patient_support_beneficial": len(set(patients[idx[beneficial]])) if n_beneficial else 0,
    }

    by_target = {}
    for c in range(N_CLASSES):
        cname = CLASS_NAMES[c]
        tmask = chosen == c
        n_acted_c = int(tmask.sum())
        true_actionable_target_total = int(np.sum((label_act == 1) & (data["transition_target_oracle"] == c)))
        if n_acted_c == 0:
            by_target[cname] = {
                "n_acted": 0, "beneficial": 0, "harmful": 0, "net_corrected": 0,
                "true_oracle_actionable_total_for_this_target": true_actionable_target_total,
                "correction_coverage_of_target_actionable": 0.0, "patient_support_beneficial": 0,
            }
            continue
        ben_c = int(beneficial[tmask].sum())
        harm_c = int(harmful[tmask].sum())
        pat_support = len(set(patients[idx[tmask][beneficial[tmask]]])) if ben_c else 0
        by_target[cname] = {
            "n_acted": n_acted_c, "beneficial": ben_c, "harmful": harm_c, "net_corrected": ben_c - harm_c,
            "beneficial_to_harmful_ratio": (ben_c / harm_c) if harm_c > 0 else (float("inf") if ben_c > 0 else None),
            "harmful_edit_rate": harm_c / n_acted_c, "correction_precision": ben_c / n_acted_c,
            "true_oracle_actionable_total_for_this_target": true_actionable_target_total,
            "correction_coverage_of_target_actionable": ben_c / true_actionable_target_total if true_actionable_target_total else None,
            "patient_support_beneficial": pat_support,
        }
    overall["by_predicted_target"] = by_target
    return overall


def apply_viability_rule(by_target_stats):
    """Section 13's predeclared rule, applied per predicted target class."""
    verdicts = {}
    for cname, v in by_target_stats.items():
        ratio = v.get("beneficial_to_harmful_ratio")
        net = v.get("net_corrected", 0)
        coverage = v.get("correction_coverage_of_target_actionable")
        patients_n = v.get("patient_support_beneficial", 0)
        passes = (
            ratio is not None and ratio > 2.0 and net > 0
            and coverage is not None and coverage >= 0.10
            and patients_n >= 10
        )
        verdicts[cname] = {"passes_viability_rule": bool(passes), "ratio": ratio, "net": net,
                            "coverage": coverage, "patients": patients_n}
    return verdicts


def main():
    import sys
    qtrans_results_path = REPO_ROOT / "PTDR_STAGE15_QTRANSITION_S2_RESULTS.json"
    assert qtrans_results_path.exists(), "run stage15_qtransition.py first"
    with open(qtrans_results_path) as f:
        qtrans = json.load(f)
    best_name = qtrans["best_variant_balanced_masked"]
    final_weighting = qtrans.get("final_weighting_carried_forward", "balanced")
    class_weight = None if final_weighting == "unweighted" else "balanced"
    model_map = {"T0": ("logreg", "S1"), "T1": ("logreg", "S2"), "T2": ("mlp", "S2")}
    model_type, feat_name = model_map[best_name]
    print(f"Using best variant {best_name} ({model_type} on {feat_name}, weighting={final_weighting})")

    data = load_all_regions_with_oracle()
    X_trans = data["S1"] if feat_name == "S1" else data["S2"]

    print("Fitting learned Q_act (S2 MLP, frozen-fold cross-fit, ALL regions)...")
    q_act_prob = learned_q_act_oof(data["S2"], data["label_act"], data["fold"])

    print("Fitting learned Q_transition (best variant, frozen-fold cross-fit, inference on ALL regions)...")
    oof_proba, permitted_sets_per_fold = learned_q_transition_oof_full_population(
        X_trans, data["label_act"], data["transition_target_oracle"], data["pred_class"], data["fold"],
        model_type, class_weight)
    masked_proba = mask_and_renormalize(oof_proba, data["pred_class"], data["fold"], permitted_sets_per_fold)
    chosen_target = masked_proba.argmax(axis=1)
    top_conf = masked_proba.max(axis=1)

    print("Sweeping act thresholds (realistic, no oracle gating)...")
    sweep = []
    for T in ACT_THRESHOLD_GRID:
        res = evaluate_controller(q_act_prob, chosen_target, data, T)
        sweep.append(res)
        if res.get("n_acted", 0) > 0:
            print(f"  T_act={T}: acted={res['n_acted']} beneficial={res['beneficial_edits']} "
                  f"harmful={res['harmful_edits']} net={res['net_corrected']} "
                  f"NETC(cov={res['by_predicted_target']['netc']['correction_coverage_of_target_actionable']}) "
                  f"RC(cov={res['by_predicted_target']['rc']['correction_coverage_of_target_actionable']})")

    # choose a representative "default" operating point: T_act=0.5 (matches K2's own 0.5 threshold convention)
    default_idx = [i for i, s in enumerate(sweep) if s["act_threshold"] == 0.5][0]
    default_result = sweep[default_idx]

    # ---- selective subset test: for NETC/RC, additionally gate on Q_transition confidence ----
    selective_sweep = {}
    for target_name, target_idx in [("netc", 1), ("rc", 4)]:
        rows = []
        for T_act in [0.3, 0.5]:
            for T_conf in [0.5, 0.6, 0.7, 0.8, 0.9]:
                res = evaluate_controller(q_act_prob, chosen_target, data, T_act, conf_threshold=T_conf, top_conf=top_conf)
                if res.get("n_acted", 0) == 0:
                    rows.append({"act_threshold": T_act, "conf_threshold": T_conf, "n_acted": 0})
                    continue
                tgt_stats = res["by_predicted_target"][target_name]
                rows.append({"act_threshold": T_act, "conf_threshold": T_conf, **tgt_stats})
        selective_sweep[target_name] = rows

    # ---- apply predeclared viability rule (section 13) ----
    viability_default = apply_viability_rule(default_result["by_predicted_target"])
    # check if ANY threshold in the full act sweep satisfies the rule for NETC/RC (RECOVERED check)
    recovered_check = {"netc": False, "rc": False}
    best_passing_config = {"netc": None, "rc": None}
    for res in sweep:
        if res.get("n_acted", 0) == 0:
            continue
        for cname in ["netc", "rc"]:
            v = apply_viability_rule({cname: res["by_predicted_target"][cname]})[cname]
            if v["passes_viability_rule"]:
                recovered_check[cname] = True
                if best_passing_config[cname] is None:
                    best_passing_config[cname] = {"act_threshold": res["act_threshold"], **v}
    # check selective (act+conf gated) subset for SELECTIVE classification
    selective_check = {"netc": False, "rc": False}
    best_selective_config = {"netc": None, "rc": None}
    for cname in ["netc", "rc"]:
        for row in selective_sweep[cname]:
            if row.get("n_acted", 0) == 0:
                continue
            v = apply_viability_rule({cname: {k: row[k] for k in row if k not in ("act_threshold", "conf_threshold")}})[cname]
            if v["passes_viability_rule"]:
                selective_check[cname] = True
                if best_selective_config[cname] is None:
                    best_selective_config[cname] = {"act_threshold": row["act_threshold"],
                                                     "conf_threshold": row["conf_threshold"], **v}

    per_target_classification = {}
    for cname in ["netc", "rc", "snfh", "et", "bg"]:
        if cname in ["netc", "rc"]:
            if recovered_check[cname]:
                verdict = "TRANSITION_SIGNAL_RECOVERED"
            elif selective_check[cname]:
                verdict = "TRANSITION_SIGNAL_SELECTIVE"
            else:
                verdict = "TRANSITION_SIGNAL_WEAK"
        else:
            v = apply_viability_rule({cname: default_result["by_predicted_target"][cname]})[cname]
            verdict = "TRANSITION_SIGNAL_RECOVERED" if v["passes_viability_rule"] else "TRANSITION_SIGNAL_WEAK"
        per_target_classification[cname] = verdict

    out = {
        "model_used": {"variant": best_name, "model_type": model_type, "feature_set": feat_name,
                        "weighting": final_weighting},
        "q_act_model": "S2 MLPClassifier(16,), frozen-fold cross-fit, ALL regions (reproduces Stage-1 K2's best model)",
        "act_threshold_sweep_no_oracle_gating": sweep,
        "default_operating_point_T_act_0.5": default_result,
        "selective_subset_sweep_netc_rc": selective_sweep,
        "viability_rule_section13": {
            "default_operating_point_viability": viability_default,
            "recovered_at_any_act_threshold": recovered_check,
            "best_passing_unrestricted_config": best_passing_config,
            "selective_subset_passes": selective_check,
            "best_passing_selective_config": best_selective_config,
        },
        "per_target_classification": per_target_classification,
        "note_oracle_assumption_removed": "Unlike the original Stage-1 K5 (which restricted the acted-upon "
            "population to oracle label_act==1 rows before applying the sweep), this script lets the LEARNED "
            "Q_act probability alone decide eligibility to act across ALL predicted-FG regions, and applies "
            "the LEARNED, source-conditioned Q_transition's top prediction (never the oracle target) at the "
            "decision point. Ground truth class_voxel_counts is used only to SCORE whether an already-applied "
            "edit was beneficial or harmful after the fact, never to decide whether/what to edit.",
    }
    (REPO_ROOT / "PTDR_STAGE15_REALISTIC_K5.json").write_text(json.dumps(out, indent=2, default=str))
    print("\nWrote PTDR_STAGE15_REALISTIC_K5.json")
    print(f"Per-target viability classification: {per_target_classification}")
    print(f"NETC default (T=0.5): {default_result['by_predicted_target']['netc']}")
    print(f"RC default (T=0.5): {default_result['by_predicted_target']['rc']}")


if __name__ == "__main__":
    main()
