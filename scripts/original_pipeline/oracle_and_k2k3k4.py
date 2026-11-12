"""PTDR Stage 1, sections 16-19: rebuild oracle action labels from genuine OOF
3D predictions (frozen Phase-0.5 definition, not reused 2D-proxy labels), then
rerun K2 (Q_act), K3 (Q_transition), K4 (S0 vs S1 vs S2 information ablation).

Cheap models only (logistic regression, small MLP), patient-grouped
GroupKFold -- same discipline as Phase 0.5, applied to real OOF 3D region
records this time.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, brier_score_loss, confusion_matrix,
    f1_score, log_loss, roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1])); from mriseg_paths import P as _P, CODE as _CODE  # noqa: E402
REPO_ROOT = _P(r"")
CLASS_NAMES = ["bg", "netc", "snfh", "et", "rc"]
N_FOLDS_EVAL = 5
RNG_SEED = 20261001
ACT_THRESHOLD_GRID = [0.3, 0.4, 0.5, 0.6, 0.7]


def build_oracle_labels(regions):
    """Identical methodology to PTDR_ORACLE_ACTION_LABEL_SPEC.md (Phase 0.5): U(a,r) =
    voxels matching class a; a*_r = argmax; KEEP-biased deterministic tie-break."""
    out = []
    for r in regions:
        counts = np.array(r["class_voxel_counts"])
        pred_c = r["pred_class"]
        max_u = counts.max()
        tied = np.flatnonzero(counts == max_u)
        a_star = pred_c if pred_c in tied else int(tied.min())
        label_act = int(a_star != pred_c)
        out.append({**r, "a_star": int(a_star), "label_act": label_act,
                    "transition_target": int(a_star) if label_act else None})
    return out


def main():
    with open(REPO_ROOT / "PTDR_3D_OOF_ALL_REGIONS.json") as f:
        regions = json.load(f)
    print(f"{len(regions)} pooled OOF 3D regions")

    oracle = build_oracle_labels(regions)
    n_actionable = sum(r["label_act"] for r in oracle)
    print(f"Oracle-actionable: {n_actionable}/{len(oracle)} ({100*n_actionable/len(oracle):.1f}%)")

    with open(REPO_ROOT / "PTDR_3D_OOF_ACTION_LABELS.json", "w") as f:
        json.dump(oracle, f)
    print("Wrote PTDR_3D_OOF_ACTION_LABELS.json")

    # ---- build feature matrix + parquet region-state table ----
    patients = np.array([r["patient"] for r in oracle])
    S0 = np.array([r["mean_probs"] for r in oracle], dtype=np.float64)  # (n,5)
    pred_onehot = np.zeros((len(oracle), 5))
    for i, r in enumerate(oracle):
        pred_onehot[i, r["pred_class"]] = 1.0
    log_size = np.log1p(np.array([r["size_voxels_mm3"] for r in oracle], dtype=np.float64)).reshape(-1, 1)
    S1 = np.concatenate([S0, pred_onehot, log_size], axis=1)
    decoder_feat_mean = np.array([r["decoder_feat_mean"] for r in oracle], dtype=np.float64)
    decoder_feat_std = np.array([r["decoder_feat_std"] for r in oracle], dtype=np.float64)
    S2 = np.concatenate([S1, decoder_feat_mean, decoder_feat_std], axis=1)

    y_act = np.array([r["label_act"] for r in oracle])
    pred_class = np.array([r["pred_class"] for r in oracle])
    transition_target = np.array([r["transition_target"] if r["transition_target"] is not None else -1 for r in oracle])

    df = pd.DataFrame({
        "patient": patients, "case": [r["case"] for r in oracle], "fold": [r["fold"] for r in oracle],
        "pred_class": pred_class, "size_voxels_mm3": [r["size_voxels_mm3"] for r in oracle],
        "label_act": y_act, "transition_target": transition_target,
        **{f"mean_prob_{c}": S0[:, i] for i, c in enumerate(CLASS_NAMES)},
        **{f"decoder_feat_mean_{i}": decoder_feat_mean[:, i] for i in range(decoder_feat_mean.shape[1])},
        **{f"decoder_feat_std_{i}": decoder_feat_std[:, i] for i in range(decoder_feat_std.shape[1])},
    })
    df.to_parquet(REPO_ROOT / "PTDR_3D_OOF_REGION_STATE.parquet", index=False)
    print(f"Wrote PTDR_3D_OOF_REGION_STATE.parquet ({len(df)} rows, {len(df.columns)} cols)")

    # ---- K2: Q_act, S0 vs S1 vs S2 ----
    def run_k2(X, name):
        gkf = GroupKFold(n_splits=N_FOLDS_EVAL)
        oof_lr = np.zeros(len(y_act)); oof_mlp = np.zeros(len(y_act))
        for tr, te in gkf.split(X, y_act, patients):
            scaler = StandardScaler().fit(X[tr])
            Xtr, Xte = scaler.transform(X[tr]), scaler.transform(X[te])
            lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xtr, y_act[tr])
            oof_lr[te] = lr.predict_proba(Xte)[:, 1]
            mlp = MLPClassifier(hidden_layer_sizes=(16,), max_iter=500, random_state=0,
                                 early_stopping=True).fit(Xtr, y_act[tr])
            oof_mlp[te] = mlp.predict_proba(Xte)[:, 1]
        results = {}
        for mname, oof in (("logreg", oof_lr), ("mlp", oof_mlp)):
            pred_bin = (oof >= 0.5).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_act, pred_bin).ravel()
            results[mname] = {
                "feature_set": name, "auroc": float(roc_auc_score(y_act, oof)),
                "auprc": float(average_precision_score(y_act, oof)),
                "brier": float(brier_score_loss(y_act, oof)),
                "nll": float(log_loss(y_act, np.clip(oof, 1e-7, 1 - 1e-7))),
                "sensitivity_at_0.5": float(tp / (tp + fn)) if (tp + fn) else None,
                "specificity_at_0.5": float(tn / (tn + fp)) if (tn + fp) else None,
            }
        # patient-grouped bootstrap CI on the stronger model's AUROC
        best_oof = oof_mlp if results["mlp"]["auroc"] >= results["logreg"]["auroc"] else oof_lr
        unique_p = np.array(sorted(set(patients)))
        rng = np.random.default_rng(RNG_SEED)
        aurocs = []
        for _ in range(1000):
            sampled = rng.choice(unique_p, size=len(unique_p), replace=True)
            idx = np.concatenate([np.flatnonzero(patients == p) for p in sampled])
            if len(set(y_act[idx])) < 2:
                continue
            aurocs.append(roc_auc_score(y_act[idx], best_oof[idx]))
        ci = [float(np.percentile(aurocs, 2.5)), float(np.percentile(aurocs, 97.5))] if aurocs else None
        return results, oof_lr, oof_mlp, ci

    print("\n=== K2 (Q_act): S0 / S1 / S2 ===")
    k2_s0, _, _, ci_s0 = run_k2(S0, "S0")
    k2_s1, oof_lr_s1, oof_mlp_s1, ci_s1 = run_k2(S1, "S1")
    k2_s2, _, _, ci_s2 = run_k2(S2, "S2")
    for name, res, ci in (("S0", k2_s0, ci_s0), ("S1", k2_s1, ci_s1), ("S2", k2_s2, ci_s2)):
        print(f"  {name}: logreg AUROC={res['logreg']['auroc']:.4f} mlp AUROC={res['mlp']['auroc']:.4f} "
              f"(best-model patient-bootstrap CI: {ci})")

    k2_out = {"S0": k2_s0, "S1": k2_s1, "S2": k2_s2,
              "S0_auroc_ci": ci_s0, "S1_auroc_ci": ci_s1, "S2_auroc_ci": ci_s2}
    (REPO_ROOT / "PTDR_3D_K2_RESULTS.json").write_text(json.dumps(k2_out, indent=2))

    # ---- K4: does S2 add value over S1? does S1 add value over S0? ----
    k4_out = {
        "S0_to_S1_logreg_delta": k2_s1["logreg"]["auroc"] - k2_s0["logreg"]["auroc"],
        "S0_to_S1_mlp_delta": k2_s1["mlp"]["auroc"] - k2_s0["mlp"]["auroc"],
        "S1_to_S2_logreg_delta": k2_s2["logreg"]["auroc"] - k2_s1["logreg"]["auroc"],
        "S1_to_S2_mlp_delta": k2_s2["mlp"]["auroc"] - k2_s1["mlp"]["auroc"],
        "verdict_S2_adds_value": (k2_s2["mlp"]["auroc"] - k2_s1["mlp"]["auroc"]) > 0.01,  # small, predeclared-here bar
    }
    print(f"\n=== K4: information ablation ===\n{json.dumps(k4_out, indent=2)}")
    (REPO_ROOT / "PTDR_3D_K4_RESULTS.json").write_text(json.dumps(k4_out, indent=2))

    # ---- K3: Q_transition, oracle-actionable only, S1 features (S2 not used for K3 per
    # scope -- K3's purpose is transition-target learnability, S0/S1/S2 comparison is K2/K4's job) ----
    print("\n=== K3 (Q_transition) ===")
    act_mask = y_act == 1
    idx_act = np.flatnonzero(act_mask)
    patients_act = patients[idx_act]
    X_act = S1[idx_act]
    pred_act = pred_class[idx_act]
    target_act = transition_target[idx_act]

    gkf = GroupKFold(n_splits=N_FOLDS_EVAL)
    oof_pred = np.full(len(target_act), -1)
    for tr, te in gkf.split(X_act, target_act, patients_act):
        clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(X_act[tr], target_act[tr])
        oof_pred[te] = clf.predict(X_act[te])

    macro_f1 = f1_score(target_act, oof_pred, average="macro", zero_division=0)
    classes_present = sorted(set(target_act))
    per_class = {}
    for c in classes_present:
        mask = target_act == c
        tp = int(np.sum((oof_pred == c) & mask))
        fp = int(np.sum((oof_pred == c) & ~mask))
        fn = int(np.sum((oof_pred != c) & mask))
        prec = tp / (tp + fp) if (tp + fp) else None
        rec = tp / (tp + fn) if (tp + fn) else None
        f1 = (2 * prec * rec / (prec + rec)) if (prec and rec and prec + rec > 0) else None
        per_class[CLASS_NAMES[c]] = {"support": int(mask.sum()), "precision": prec, "recall": rec, "f1": f1}

    cm = confusion_matrix(target_act, oof_pred, labels=classes_present).tolist()
    patient_coverage = len(set(patients_act)) / len(set(patients))
    print(f"  macro-F1={macro_f1:.4f}, patient coverage={patient_coverage:.3f}")
    for cname, v in per_class.items():
        print(f"    {cname}: support={v['support']} P={v['precision']} R={v['recall']} F1={v['f1']}")

    k3_out = {"n_actionable": len(idx_act), "macro_f1": float(macro_f1), "per_class": per_class,
              "confusion_matrix": cm, "confusion_matrix_labels": [CLASS_NAMES[c] for c in classes_present],
              "patient_coverage": patient_coverage}
    (REPO_ROOT / "PTDR_3D_K3_RESULTS.json").write_text(json.dumps(k3_out, indent=2))
    print("\nWrote PTDR_3D_K2/K3/K4_RESULTS.json")


if __name__ == "__main__":
    main()
