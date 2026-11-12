"""PTDR Stage 1.5, sections 5-11: Q_transition diagnostic closure.

Uses ONLY the frozen oracle-actionable (label_act=1) predicted-FG regions and
their already-extracted S1/S2 features. Uses the RECORDED patient-grouped
`fold` column (0-4) from PTDR_3D_OOF_ACTION_LABELS.json as the outer CV split
throughout (NOT a freshly-derived GroupKFold, per the governing task's explicit
instruction and per PTDR_STAGE15_K3_IMPLEMENTATION_AUDIT.md's finding that the
original K3 used its own independent GroupKFold partition).

Model variants (cheap sklearn only, no architecture search):
  T0 = S1 LogisticRegression
  T1 = S2 LogisticRegression
  T2 = S2 MLPClassifier(16,16)
Each evaluated UNMASKED (replicates original K3 semantics with the frozen-fold
CV swap) and MASKED (source-conditioned: permitted target set for source class
i is derived from TRAINING-FOLD-ONLY occurrence, logits renormalized over the
permitted set before argmax).

Outputs:
  PTDR_STAGE15_QTRANSITION_S2_RESULTS.json  (sections 6-7, all 6 variants + calibration for best)
  PTDR_STAGE15_IMBALANCE_CONTROL.json       (section 8)
  PTDR_STAGE15_PR_CURVES.json               (section 9)
  PTDR_STAGE15_SELECTIVE_TRANSITION.json    (section 10)
"""
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, balanced_accuracy_score, confusion_matrix,
    f1_score, log_loss, precision_recall_curve,
)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1])); from mriseg_paths import P as _P, CODE as _CODE  # noqa: E402
REPO_ROOT = _P(r"")
CLASS_NAMES = ["bg", "netc", "snfh", "et", "rc"]
N_CLASSES = 5
N_FOLDS = 5
RNG_SEED = 20261001


def load_actionable():
    with open(REPO_ROOT / "PTDR_3D_OOF_ACTION_LABELS.json") as f:
        oracle = json.load(f)
    act = [r for r in oracle if r["label_act"] == 1]
    patients = np.array([r["patient"] for r in act])
    fold = np.array([r["fold"] for r in act])
    pred_class = np.array([r["pred_class"] for r in act])
    target = np.array([r["transition_target"] for r in act])
    class_voxel_counts = np.array([r["class_voxel_counts"] for r in act], dtype=np.float64)

    S0 = np.array([r["mean_probs"] for r in act], dtype=np.float64)
    pred_onehot = np.zeros((len(act), 5))
    for i, r in enumerate(act):
        pred_onehot[i, r["pred_class"]] = 1.0
    log_size = np.log1p(np.array([r["size_voxels_mm3"] for r in act], dtype=np.float64)).reshape(-1, 1)
    S1 = np.concatenate([S0, pred_onehot, log_size], axis=1)
    decoder_feat_mean = np.array([r["decoder_feat_mean"] for r in act], dtype=np.float64)
    decoder_feat_std = np.array([r["decoder_feat_std"] for r in act], dtype=np.float64)
    S2 = np.concatenate([S1, decoder_feat_mean, decoder_feat_std], axis=1)
    return {
        "patients": patients, "fold": fold, "pred_class": pred_class, "target": target,
        "class_voxel_counts": class_voxel_counts, "S1": S1, "S2": S2, "n": len(act),
    }


def align_proba(proba, classes_, n_classes=N_CLASSES):
    """Map sklearn's predict_proba columns (ordered by clf.classes_, which may omit
    classes absent from the training fold) onto a fixed 0..n_classes-1 column order."""
    out = np.zeros((proba.shape[0], n_classes))
    for j, c in enumerate(classes_):
        out[:, int(c)] = proba[:, j]
    return out


def fit_cv_oof_proba(X, y, fold_arr, model_type, class_weight):
    """Cross-fitted OOF probability matrix (n, 5) using the RECORDED fold column
    as the train/test partition (5 folds, patient-grouped by construction)."""
    n = len(y)
    oof_proba = np.zeros((n, N_CLASSES))
    for k in range(N_FOLDS):
        te = fold_arr == k
        tr = ~te
        scaler = StandardScaler().fit(X[tr])
        Xtr, Xte = scaler.transform(X[tr]), scaler.transform(X[te])
        if model_type == "logreg":
            clf = LogisticRegression(max_iter=2000, class_weight=class_weight).fit(Xtr, y[tr])
        elif model_type == "mlp":
            clf = MLPClassifier(hidden_layer_sizes=(16, 16), max_iter=500, random_state=0,
                                 early_stopping=True).fit(Xtr, y[tr])
        else:
            raise ValueError(model_type)
        proba = clf.predict_proba(Xte)
        oof_proba[te] = align_proba(proba, clf.classes_)
    return oof_proba


def build_permitted_sets_per_fold(pred_class_arr, target_arr, fold_arr):
    """Section 7: per outer fold, permitted target set for each source class i is
    derived ONLY from training-fold occurrence (which target classes j actually occur
    for source i among the training patients' rows) -- never from the held-out fold."""
    permitted = {}  # fold -> {source_class: set(target_classes)}
    for k in range(N_FOLDS):
        tr = fold_arr != k
        this_fold = {}
        for src in sorted(set(pred_class_arr[tr])):
            mask = tr & (pred_class_arr == src)
            this_fold[int(src)] = set(int(t) for t in target_arr[mask])
        permitted[k] = this_fold
    return permitted


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


def patient_bootstrap_macrof1_ci(y_true, y_pred, patients, n_boot=1000, seed=RNG_SEED):
    unique_p = np.array(sorted(set(patients)))
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        sampled = rng.choice(unique_p, size=len(unique_p), replace=True)
        idx = np.concatenate([np.flatnonzero(patients == p) for p in sampled])
        if len(set(y_true[idx])) < 2:
            continue
        vals.append(f1_score(y_true[idx], y_pred[idx], average="macro", zero_division=0))
    if not vals:
        return None
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def brier_multiclass(y_true, proba, n_classes=N_CLASSES):
    """One-vs-rest averaged Brier score."""
    briers = []
    for c in range(n_classes):
        ind = (y_true == c).astype(float)
        briers.append(float(np.mean((proba[:, c] - ind) ** 2)))
    return float(np.mean(briers)), briers


def evaluate_variant(y_true, proba, pred_class_arr, patients, name):
    y_pred = np.argmax(proba, axis=1)
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    proba_clipped = np.clip(proba, 1e-12, 1.0)
    proba_clipped = proba_clipped / proba_clipped.sum(axis=1, keepdims=True)
    nll = float(log_loss(y_true, proba_clipped, labels=list(range(N_CLASSES))))
    brier_avg, brier_per_class = brier_multiclass(y_true, proba)

    per_class = {}
    for c in range(N_CLASSES):
        mask = y_true == c
        tp = int(np.sum((y_pred == c) & mask))
        fp = int(np.sum((y_pred == c) & ~mask))
        fn = int(np.sum((y_pred != c) & mask))
        prec = tp / (tp + fp) if (tp + fp) else None
        rec = tp / (tp + fn) if (tp + fn) else None
        f1 = (2 * prec * rec / (prec + rec)) if (prec and rec and prec + rec > 0) else None
        per_class[CLASS_NAMES[c]] = {"support": int(mask.sum()), "precision": prec, "recall": rec, "f1": f1}

    cm = confusion_matrix(y_true, y_pred, labels=list(range(N_CLASSES))).tolist()
    ci = patient_bootstrap_macrof1_ci(y_true, y_pred, patients)

    by_source = {}
    for src in sorted(set(pred_class_arr)):
        m = pred_class_arr == src
        if m.sum() == 0:
            continue
        f1_src = float(f1_score(y_true[m], y_pred[m], average="macro", zero_division=0))
        per_class_src = {}
        for c in sorted(set(y_true[m])):
            mask = (y_true == c) & m
            tp = int(np.sum((y_pred == c) & mask))
            fp = int(np.sum((y_pred == c) & m & ~(y_true == c)))
            fn = int(np.sum((y_pred != c) & mask))
            prec = tp / (tp + fp) if (tp + fp) else None
            rec = tp / (tp + fn) if (tp + fn) else None
            per_class_src[CLASS_NAMES[c]] = {"support": int(mask.sum()), "precision": prec, "recall": rec}
        by_source[CLASS_NAMES[src]] = {"n_regions": int(m.sum()), "macro_f1": f1_src, "per_target": per_class_src}

    return {
        "name": name, "n": int(len(y_true)), "macro_f1": macro_f1, "balanced_accuracy": bal_acc,
        "nll": nll, "brier_avg": brier_avg, "brier_per_class": dict(zip(CLASS_NAMES, brier_per_class)),
        "per_class": per_class, "confusion_matrix": cm, "confusion_matrix_labels": CLASS_NAMES,
        "macro_f1_patient_bootstrap_ci": ci, "by_source_class": by_source,
    }, y_pred


def ece_adaptive_ece(y_true, proba, n_bins=12):
    conf = proba.max(axis=1)
    pred = proba.argmax(axis=1)
    correct = (pred == y_true).astype(float)
    # equal-width ECE
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    reliability = []
    n = len(conf)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if mask.sum() == 0:
            reliability.append({"bin_lo": float(lo), "bin_hi": float(hi), "n": 0,
                                 "mean_confidence": None, "empirical_accuracy": None})
            continue
        mean_conf = float(conf[mask].mean())
        emp_acc = float(correct[mask].mean())
        ece += (mask.sum() / n) * abs(mean_conf - emp_acc)
        reliability.append({"bin_lo": float(lo), "bin_hi": float(hi), "n": int(mask.sum()),
                             "mean_confidence": mean_conf, "empirical_accuracy": emp_acc})
    # adaptive (equal-frequency) ECE
    order = np.argsort(conf)
    chunks = np.array_split(order, n_bins)
    adaptive_ece = 0.0
    for idx in chunks:
        if len(idx) == 0:
            continue
        mean_conf = conf[idx].mean()
        emp_acc = correct[idx].mean()
        adaptive_ece += (len(idx) / n) * abs(mean_conf - emp_acc)
    return float(ece), float(adaptive_ece), reliability


def temperature_scale_cv(raw_oof_proba, y_true, fold_arr, pred_class_arr, permitted_sets_per_fold):
    """Section 11: per-outer-fold temperature scaling, T fit on TRAINING FOLD data only
    (grid search minimizing training-fold NLL on log-probabilities), applied to the held-out
    fold's probabilities, then re-masked/renormalized with the same source-conditioned mask."""
    log_proba = np.log(np.clip(raw_oof_proba, 1e-12, 1.0))
    calibrated = np.zeros_like(raw_oof_proba)
    temps = {}
    grid = np.linspace(0.3, 3.0, 28)
    for k in range(N_FOLDS):
        tr = fold_arr != k
        te = fold_arr == k
        best_T, best_nll = 1.0, np.inf
        for T in grid:
            scaled = log_proba[tr] / T
            scaled = scaled - scaled.max(axis=1, keepdims=True)
            p = np.exp(scaled)
            p = p / p.sum(axis=1, keepdims=True)
            nll = log_loss(y_true[tr], np.clip(p, 1e-12, 1.0), labels=list(range(N_CLASSES)))
            if nll < best_nll:
                best_nll, best_T = nll, T
        temps[k] = float(best_T)
        scaled = log_proba[te] / best_T
        scaled = scaled - scaled.max(axis=1, keepdims=True)
        p = np.exp(scaled)
        p = p / p.sum(axis=1, keepdims=True)
        calibrated[te] = p
    calibrated_masked = mask_and_renormalize(calibrated, pred_class_arr, fold_arr, permitted_sets_per_fold)
    return calibrated_masked, temps


def main():
    data = load_actionable()
    y = data["target"]
    patients = data["patients"]
    fold_arr = data["fold"]
    pred_class_arr = data["pred_class"]
    S1, S2 = data["S1"], data["S2"]
    n = data["n"]
    print(f"Loaded {n} oracle-actionable regions, fold counts: {np.bincount(fold_arr)}")

    permitted_sets_per_fold = build_permitted_sets_per_fold(pred_class_arr, y, fold_arr)
    for k, v in permitted_sets_per_fold.items():
        for src, tgts in v.items():
            expected_no_self = set(range(N_CLASSES)) - {src}
            if tgts != expected_no_self and len(tgts) > 0:
                print(f"  NOTE fold{k} source={CLASS_NAMES[src]}: permitted targets = "
                      f"{[CLASS_NAMES[t] for t in sorted(tgts)]} (differs from 'all except self')")

    variants_spec = [
        ("T0", "logreg", S1), ("T1", "logreg", S2), ("T2", "mlp", S2),
    ]

    all_results = {}
    raw_proba_store = {}  # (name, weighting) -> oof_proba (unmasked, balanced only, for reuse)
    for vname, model_type, X in variants_spec:
        for weighting in ["balanced"]:  # sections 6-7 use the original-matching balanced weighting
            oof_proba = fit_cv_oof_proba(X, y, fold_arr, model_type, weighting)
            raw_proba_store[(vname, weighting)] = oof_proba
            unmasked_res, _ = evaluate_variant(y, oof_proba, pred_class_arr, patients, f"{vname}_unmasked_{weighting}")
            masked_proba = mask_and_renormalize(oof_proba, pred_class_arr, fold_arr, permitted_sets_per_fold)
            masked_res, _ = evaluate_variant(y, masked_proba, pred_class_arr, patients, f"{vname}_masked_{weighting}")
            all_results[vname] = all_results.get(vname, {})
            all_results[vname]["unmasked"] = unmasked_res
            all_results[vname]["masked"] = masked_res
            delta = {
                "macro_f1_delta_masked_minus_unmasked": masked_res["macro_f1"] - unmasked_res["macro_f1"],
                "netc_precision_unmasked": unmasked_res["per_class"]["netc"]["precision"],
                "netc_precision_masked": masked_res["per_class"]["netc"]["precision"],
                "rc_precision_unmasked": unmasked_res["per_class"]["rc"]["precision"],
                "rc_precision_masked": masked_res["per_class"]["rc"]["precision"],
            }
            all_results[vname]["masking_delta"] = delta
            print(f"{vname} ({weighting}): unmasked macro_f1={unmasked_res['macro_f1']:.4f} "
                  f"masked macro_f1={masked_res['macro_f1']:.4f} | "
                  f"NETC prec {delta['netc_precision_unmasked']}->{delta['netc_precision_masked']} | "
                  f"RC prec {delta['rc_precision_unmasked']}->{delta['rc_precision_masked']}")

    # ---- pick best masked variant (sections 7-11's "best T-variant") ----
    complexity_rank = {"T0": 0, "T1": 1, "T2": 2}
    best_name = max(all_results.keys(),
                     key=lambda v: (round(all_results[v]["masked"]["macro_f1"], 6), -complexity_rank[v]))
    print(f"\nBest masked variant (by macro-F1, ties->simpler): {best_name} "
          f"(macro_f1={all_results[best_name]['masked']['macro_f1']:.4f})")

    out6_7 = {
        "variants": all_results,
        "best_variant_balanced_masked": best_name,
        "selection_rule": "argmax macro_f1 among masked T0/T1/T2 (balanced class_weight); ties broken toward simpler model",
        "note_source_conditioned_masking": "Permitted target sets were derived per outer fold from TRAINING "
            "data only (never held-out fold labels); empirically every fold's permitted set reduced to "
            "'all 5 classes except the region's own source class' -- see PTDR_STAGE15_TRANSITION_MATRIX.md "
            "section 5 (16/20 possible (source,target) pairs observed; the missing 4 are exactly the "
            "4 structurally-impossible self-transitions). Masking therefore has one concrete, well-defined "
            "effect here: it prevents the classifier from ever predicting a region's own source class as "
            "its correction target (a prediction that is guaranteed wrong for every oracle-actionable row "
            "by construction) -- it does not further restrict the output space beyond that.",
    }

    # ---- Section 8: imbalance control on the best variant ----
    best_vname, best_model_type, best_X = {v[0]: v for v in variants_spec}[best_name]
    oof_balanced = raw_proba_store[(best_name, "balanced")]
    oof_unweighted = fit_cv_oof_proba(best_X, y, fold_arr, best_model_type, None)
    masked_balanced = mask_and_renormalize(oof_balanced, pred_class_arr, fold_arr, permitted_sets_per_fold)
    masked_unweighted = mask_and_renormalize(oof_unweighted, pred_class_arr, fold_arr, permitted_sets_per_fold)
    res_balanced, _ = evaluate_variant(y, masked_balanced, pred_class_arr, patients, f"{best_name}_masked_balanced")
    res_unweighted, _ = evaluate_variant(y, masked_unweighted, pred_class_arr, patients, f"{best_name}_masked_unweighted")
    imbalance_out = {
        "best_variant": best_name, "arm_A_standard_unweighted": res_unweighted, "arm_B_balanced": res_balanced,
        "delta_B_minus_A": {
            "macro_f1": res_balanced["macro_f1"] - res_unweighted["macro_f1"],
            "netc_precision": (res_balanced["per_class"]["netc"]["precision"] or 0) - (res_unweighted["per_class"]["netc"]["precision"] or 0),
            "rc_precision": (res_balanced["per_class"]["rc"]["precision"] or 0) - (res_unweighted["per_class"]["rc"]["precision"] or 0),
        },
        "note": "Original frozen K3 used class_weight='balanced' (confirmed in PTDR_STAGE15_K3_IMPLEMENTATION_AUDIT.md); "
                "arm B replicates that choice, arm A is the new unweighted comparison arm, both using the "
                "source-conditioned masked best variant from sections 6-7.",
    }
    (REPO_ROOT / "PTDR_STAGE15_IMBALANCE_CONTROL.json").write_text(json.dumps(imbalance_out, indent=2))
    print(f"\nSection 8 imbalance control: balanced macro_f1={res_balanced['macro_f1']:.4f} "
          f"vs unweighted macro_f1={res_unweighted['macro_f1']:.4f}")
    print(f"  NETC precision: balanced={res_balanced['per_class']['netc']['precision']} "
          f"unweighted={res_unweighted['per_class']['netc']['precision']}")
    print(f"  RC precision: balanced={res_balanced['per_class']['rc']['precision']} "
          f"unweighted={res_unweighted['per_class']['rc']['precision']}")

    # choose final weighting to carry forward into sections 9-11 (and realistic K5):
    # prefer balanced unless unweighted is clearly better on BOTH macro_f1 and NETC+RC precision
    unweighted_clearly_better = (
        res_unweighted["macro_f1"] >= res_balanced["macro_f1"]
        and (res_unweighted["per_class"]["netc"]["precision"] or 0) >= (res_balanced["per_class"]["netc"]["precision"] or 0)
        and (res_unweighted["per_class"]["rc"]["precision"] or 0) >= (res_balanced["per_class"]["rc"]["precision"] or 0)
    )
    final_weighting = "unweighted" if unweighted_clearly_better else "balanced"
    final_masked_proba = masked_unweighted if final_weighting == "unweighted" else masked_balanced
    print(f"  -> carrying forward weighting='{final_weighting}' for sections 9-11 / realistic K5")
    out6_7["final_weighting_carried_forward"] = final_weighting

    (REPO_ROOT / "PTDR_STAGE15_QTRANSITION_S2_RESULTS.json").write_text(json.dumps(out6_7, indent=2, default=str))
    print("Wrote PTDR_STAGE15_QTRANSITION_S2_RESULTS.json (updated with final_weighting; calibration appended below)")

    # ---- Section 9: PR curves (one-vs-rest) using best variant + final weighting ----
    pr_out = {}
    n_actionable = len(y)
    for c in range(N_CLASSES):
        ind = (y == c).astype(int)
        prevalence = float(ind.mean())
        scores = final_masked_proba[:, c]
        auprc = float(average_precision_score(ind, scores)) if ind.sum() > 0 else None
        precision, recall, thresholds = precision_recall_curve(ind, scores)
        # subsample curve to ~30 points across recall grid
        recall_grid = np.linspace(0, 1, 26)
        curve_points = []
        order = np.argsort(recall)
        r_sorted, p_sorted = recall[order], precision[order]
        for rg in recall_grid:
            j = np.searchsorted(r_sorted, rg)
            j = min(j, len(p_sorted) - 1)
            curve_points.append({"recall": float(rg), "precision": float(p_sorted[j])})
        pr_out[CLASS_NAMES[c]] = {
            "prevalence": prevalence, "auprc": auprc,
            "auprc_over_prevalence_ratio": (auprc / prevalence) if (auprc is not None and prevalence > 0) else None,
            "curve_points_recall_precision": curve_points,
        }
    pr_out["interpretation"] = {
        "netc_ratio": pr_out["netc"]["auprc_over_prevalence_ratio"],
        "rc_ratio": pr_out["rc"]["auprc_over_prevalence_ratio"],
        "guidance": "AUPRC far above prevalence despite low argmax precision suggests usable ranking signal "
                    "exists even though forced top-1 choice is often wrong; AUPRC near prevalence means the "
                    "representation itself lacks target information for that class.",
    }
    (REPO_ROOT / "PTDR_STAGE15_PR_CURVES.json").write_text(json.dumps(pr_out, indent=2))
    print(f"\nSection 9 PR curves: NETC AUPRC={pr_out['netc']['auprc']:.4f} (prevalence={pr_out['netc']['prevalence']:.4f}, "
          f"ratio={pr_out['netc']['auprc_over_prevalence_ratio']:.2f}) | "
          f"RC AUPRC={pr_out['rc']['auprc']:.4f} (prevalence={pr_out['rc']['prevalence']:.4f}, "
          f"ratio={pr_out['rc']['auprc_over_prevalence_ratio']:.2f})")

    # ---- Section 10: selective / abstention test ----
    top_conf = final_masked_proba.max(axis=1)
    top_pred = final_masked_proba.argmax(axis=1)
    sorted_proba = np.sort(final_masked_proba, axis=1)
    margin = sorted_proba[:, -1] - sorted_proba[:, -2]
    with np.errstate(divide="ignore", invalid="ignore"):
        entropy = -np.sum(np.where(final_masked_proba > 0, final_masked_proba * np.log(final_masked_proba + 1e-12), 0.0), axis=1)
    cvc = data["class_voxel_counts"]
    harmful = np.array([cvc[i, top_pred[i]] < cvc[i, pred_class_arr[i]] for i in range(n_actionable)])
    correct = top_pred == y
    n_patients_total = len(set(patients))

    def subset_stats(mask_group, label):
        out_thr = []
        for T in [0.5, 0.6, 0.7, 0.8, 0.9]:
            covered = mask_group & (top_conf >= T)
            n_cov = int(covered.sum())
            if n_cov == 0:
                out_thr.append({"threshold": T, "coverage": 0.0, "n_covered": 0, "precision": None,
                                 "harmful_rate": None, "patient_coverage": 0.0})
                continue
            denom = int(mask_group.sum())
            out_thr.append({
                "threshold": T, "n_covered": n_cov,
                "coverage": n_cov / denom if denom else None,
                "precision": float(correct[covered].mean()),
                "harmful_rate": float(harmful[covered].mean()),
                "patient_coverage": len(set(patients[covered])) / n_patients_total,
            })
        return out_thr

    overall_mask = np.ones(n_actionable, dtype=bool)
    netc_pred_mask = top_pred == 1
    rc_pred_mask = top_pred == 4

    def descriptive_curve(values, name):
        pct = np.percentile(values, [0, 10, 25, 50, 75, 90, 100])
        return {"percentiles_0_10_25_50_75_90_100": pct.tolist()}

    selective_out = {
        "final_weighting_used": final_weighting, "best_variant": best_name,
        "overall": subset_stats(overall_mask, "overall"),
        "predicted_target_netc": subset_stats(netc_pred_mask, "netc"),
        "predicted_target_rc": subset_stats(rc_pred_mask, "rc"),
        "margin_descriptive": descriptive_curve(margin, "margin"),
        "entropy_descriptive": descriptive_curve(entropy, "entropy"),
        "margin_descriptive_netc_pred": descriptive_curve(margin[netc_pred_mask], "margin_netc") if netc_pred_mask.sum() else None,
        "margin_descriptive_rc_pred": descriptive_curve(margin[rc_pred_mask], "margin_rc") if rc_pred_mask.sum() else None,
        "n_predicted_netc": int(netc_pred_mask.sum()), "n_predicted_rc": int(rc_pred_mask.sum()),
        "n_total_actionable": n_actionable,
    }
    (REPO_ROOT / "PTDR_STAGE15_SELECTIVE_TRANSITION.json").write_text(json.dumps(selective_out, indent=2))
    print("\nSection 10 selective transition (overall, at T=0.7):",
          json.dumps([x for x in selective_out["overall"] if x["threshold"] == 0.7], indent=2))

    # ---- Section 11: calibration for the best simple model (final weighting, masked) ----
    nll_final = float(log_loss(y, np.clip(final_masked_proba, 1e-12, 1.0), labels=list(range(N_CLASSES))))
    brier_final, brier_per_class_final = brier_multiclass(y, final_masked_proba)
    ece, adaptive_ece, reliability = ece_adaptive_ece(y, final_masked_proba, n_bins=12)
    calibrated_masked, temps = temperature_scale_cv(
        raw_proba_store[(best_name, "balanced")] if final_weighting == "balanced" else oof_unweighted,
        y, fold_arr, pred_class_arr, permitted_sets_per_fold)
    nll_ts = float(log_loss(y, np.clip(calibrated_masked, 1e-12, 1.0), labels=list(range(N_CLASSES))))
    brier_ts, _ = brier_multiclass(y, calibrated_masked)
    ece_ts, adaptive_ece_ts, reliability_ts = ece_adaptive_ece(y, calibrated_masked, n_bins=12)

    calibration_out = {
        "model": best_name, "weighting": final_weighting,
        "uncalibrated": {"nll": nll_final, "brier_avg": brier_final, "ece": ece, "adaptive_ece": adaptive_ece,
                          "reliability_bins": reliability},
        "temperature_scaled": {"per_fold_temperatures": temps, "nll": nll_ts, "brier_avg": brier_ts,
                                "ece": ece_ts, "adaptive_ece": adaptive_ece_ts, "reliability_bins": reliability_ts},
        "note": "Temperature fit per outer fold by grid search minimizing NLL on that fold's TRAINING data "
                "only (never the held-out decision fold), then applied to the held-out fold and re-masked/"
                "renormalized with the same source-conditioned mask.",
    }
    out6_7["calibration_section11"] = calibration_out
    (REPO_ROOT / "PTDR_STAGE15_QTRANSITION_S2_RESULTS.json").write_text(json.dumps(out6_7, indent=2, default=str))
    print(f"\nSection 11 calibration: uncalibrated NLL={nll_final:.4f} ECE={ece:.4f} | "
          f"temp-scaled NLL={nll_ts:.4f} ECE={ece_ts:.4f}")
    print("Final: wrote PTDR_STAGE15_QTRANSITION_S2_RESULTS.json (full), "
          "PTDR_STAGE15_IMBALANCE_CONTROL.json, PTDR_STAGE15_PR_CURVES.json, PTDR_STAGE15_SELECTIVE_TRANSITION.json")

    return {
        "best_name": best_name, "final_weighting": final_weighting,
        "final_masked_proba": final_masked_proba, "y": y, "pred_class_arr": pred_class_arr,
        "patients": patients, "fold_arr": fold_arr, "permitted_sets_per_fold": permitted_sets_per_fold,
    }


if __name__ == "__main__":
    main()
