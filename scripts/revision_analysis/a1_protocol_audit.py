"""R1.5 / R2.5 / R1.6: why does action-head AUROC fall from 0.947 to 0.588 (and Brier 0.564)?
Tests four candidate explanations with the same data:
  (a) cross-base-network feature mismatch (S2 decoder channels not aligned across the 5 networks),
  (b) joint-training interference (act-only head with the identical trunk vs Model T),
  (c) optimisation / early-stopping artefact (act-only BCE head, own early stopping),
  (d) label problem (same labels give 0.945 within a network).
"""
import json
import sys
import time

import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1])); from mriseg_paths import P as _P, CODE as _CODE  # noqa: E402
sys.path.insert(0, str(_CODE / "revision_analysis"))
from rev_common import *  # noqa
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.neural_network import MLPClassifier

d = load_dev()
y, fold, pat, X2 = d["label_act"], d["fold"], d["patients"], d["S2"]
inner, _ = p2_inner_assignment(d)
res = {"n_regions": int(d["n"]), "act_prevalence": float(y.mean())}


def mlp(Xtr, ytr, Xte):
    sc = StandardScaler().fit(Xtr)
    m = MLPClassifier((16,), max_iter=500, random_state=0, early_stopping=True).fit(sc.transform(Xtr), ytr)
    return m.predict_proba(sc.transform(Xte))[:, 1]


def patient_boot_auc(y_, s_, pats, n=500, seed=1):
    up = np.array(sorted(set(pats))); rng = np.random.default_rng(seed)
    by = {p: np.flatnonzero(pats == p) for p in up}
    a = []
    for _ in range(n):
        idx = np.concatenate([by[p] for p in rng.choice(up, len(up))])
        if len(set(y_[idx])) == 2:
            a.append(roc_auc_score(y_[idx], s_[idx]))
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]


# ---- (a) network-to-network transfer matrix: rows = training network g, cols = test network f
for fs in ("S1", "S2"):
    X = features(d, fs)
    M = np.zeros((5, 5))
    for f in range(5):                       # test network
        te_all = fold == f
        for g in range(5):                   # train network
            if g == f:                       # within-network, patient-grouped inner CV
                oof = np.zeros(d["n"])
                for j in range(N_INNER):
                    tr = te_all & (inner != j); te = te_all & (inner == j)
                    oof[te] = mlp(X[tr], y[tr], X[te])
                M[g, f] = roc_auc_score(y[te_all], oof[te_all])
            else:
                M[g, f] = roc_auc_score(y[te_all], mlp(X[fold == g], y[fold == g], X[te_all]))
    diag = float(np.mean(np.diag(M))); off = float(np.mean(M[~np.eye(5, dtype=bool)]))
    res[f"transfer_matrix_{fs}"] = {"matrix_train_row_test_col": M.round(3).tolist(),
                                    "mean_within_network": diag, "mean_cross_network": off}
    print(fs, "within-network mean AUROC", round(diag, 3), "cross-network mean", round(off, 3), flush=True)

# ---- (a') protocol comparison with the original sklearn S2/S1 MLP
prot = {}
for fs in ("S1", "S2"):
    X = features(d, fs)
    o0 = np.zeros(d["n"])                    # P0: frozen-fold cross-network
    for k in range(5):
        o0[fold == k] = mlp(X[fold != k], y[fold != k], X[fold == k])
    o2 = np.zeros(d["n"])                    # P2: within-network nested
    for f in range(5):
        for j in range(N_INNER):
            tr = (fold == f) & (inner != j); te = (fold == f) & (inner == j)
            o2[te] = mlp(X[tr], y[tr], X[te])
    for name, o in (("P0_cross_network", o0), ("P2_within_network", o2)):
        prot[f"{fs}_{name}"] = {"auroc": float(roc_auc_score(y, o)), "auroc_ci95": patient_boot_auc(y, o, pat),
                                "brier": float(brier_score_loss(y, o))}
        print(fs, name, prot[f"{fs}_{name}"], flush=True)
res["sklearn_mlp16_protocol_comparison"] = prot

# ---- (b,c) torch ablation: act-only head vs joint Model T (identical trunk/features/optimiser/early stopping)
abl = {}
X = features(d, "S2")
npz0 = np.load(REPO / "PTDR_K6_OOF_PREDICTIONS.npz") if VARIANT != "slide" else np.load(OUT / "oof_P0_unw.npz")   # original joint T under P0
abl["P0_joint_T_original_saved"] = {
    "auroc_pooled_over_seeds": float(roc_auc_score(y, np.mean([npz0[f"T_p_act_seed{s}"] for s in SEEDS], 0))),
    "auroc_per_seed": [float(roc_auc_score(y, npz0[f"T_p_act_seed{s}"])) for s in SEEDS]}
oofs = {"P0_act_only": [], "P2_act_only": []}
for s in SEEDS:
    o0 = np.zeros(d["n"])
    for tag, tr, te in iter_splits(d, "P0"):
        o0[te] = fit_predict_act_only(X, y, pat, tr, te, s, tag)
    o2 = np.zeros(d["n"])
    for tag, tr, te in iter_splits(d, "P2"):
        o2[te] = fit_predict_act_only(X, y, pat, tr, te, s, tag)
    oofs["P0_act_only"].append(o0); oofs["P2_act_only"].append(o2)
    print("act-only seed", s, "done", flush=True)
for k, v in oofs.items():
    abl[k] = {"auroc_pooled_over_seeds": float(roc_auc_score(y, np.mean(v, 0))),
              "auroc_per_seed": [float(roc_auc_score(y, a)) for a in v],
              "brier_pooled": float(brier_score_loss(y, np.mean(v, 0)))}
res["torch_ablation_S2"] = abl
res["_joint_T_P2_added_by_a3"] = "see a3_compare_models.py (joint T AUROC under P2)"
(OUT / "A1_PROTOCOL_AUDIT.json").write_text(json.dumps(res, indent=1))
print(json.dumps(abl, indent=1))
print("DONE")
