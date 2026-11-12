"""Shared helpers for the Revision-1 analyses (Paper 1, Discover Computer Vision).

Everything here reuses the frozen definitions of the original study: region table
(PTDR_3D_OOF_ACTION_LABELS.json), oracle utility, patient-level normalised net utility,
patient percentile bootstrap, and the K6 trunk/head architectures. Nothing here changes
a definition; it only adds protocols, weighting and operating-point sweeps.
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1])); from mriseg_paths import P as _P, CODE as _CODE  # noqa: E402
REPO = _P(r"")
REV = _P("Paper1_Revision/revision_analysis")
VARIANT = os.environ.get("REV_VARIANT", "slide")          # "slide" = primary (Gaussian sliding window); "tiled" = as-submitted pipeline
OUT = REV / ("outputs_slide" if VARIANT == "slide" else "outputs")
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(_CODE / "original_pipeline"))
from k6_data import load_k6_data  # noqa: E402
from k6_models import ModelT, ModelM, count_trainable_params  # noqa: E402

CLASS_NAMES = ["bg", "netc", "snfh", "et", "rc"]
N_BOOT = 2000
BOOT_SEED = 20261002
LR, MAX_EPOCHS, PATIENCE, VAL_FRAC = 1e-3, 2000, 50, 0.15
SEEDS = [0, 1, 2]
torch.set_num_threads(4)


def region_arrays_from_records(rs):
    """Region table (features, oracle labels) from freshly extracted region records."""
    n = len(rs)
    S0 = np.array([r["mean_probs"] for r in rs]); pc = np.array([r["pred_class"] for r in rs])
    oh = np.zeros((n, 5)); oh[np.arange(n), pc] = 1
    size = np.array([r["size_voxels_mm3"] for r in rs], float)
    S1 = np.concatenate([S0, oh, np.log1p(size)[:, None]], 1)
    S2 = np.concatenate([S1, np.array([r["decoder_feat_mean"] for r in rs]), np.array([r["decoder_feat_std"] for r in rs])], 1)
    cvc = np.array([r["class_voxel_counts"] for r in rs], float)
    tied = cvc == cvc.max(1, keepdims=True)
    a_star = np.where(tied[np.arange(n), pc], pc, tied.argmax(1))          # keep-biased, lowest-index tie-break
    label_act = (a_star != pc).astype(int)
    tt = np.where(label_act == 1, a_star, -1)
    return {"n": n, "patients": np.array([r["patient"] for r in rs]), "cases": np.array([r["case"] for r in rs]),
            "fold": np.array([r["base_network"] for r in rs]), "pred_class": pc, "label_act": label_act,
            "transition_target": tt, "y_final": np.where(label_act == 0, pc, tt).astype(np.int64),
            "size_voxels": size, "class_voxel_counts": cvc, "S2": S2, "n_label_violations": 0}


def load_dev():
    if VARIANT == "slide":
        recs = []
        for f in range(5):
            recs += json.loads((OUT / "dev" / f"fold{f}" / "regions.json").read_text())
        return region_arrays_from_records(recs)
    return load_k6_data()


def features(d, name):
    S2 = d["S2"]
    return {"S0": S2[:, :5], "S1": S2[:, :11], "S2": S2}[name]


# ---------------------------------------------------------------- utility / statistics
def delta_u(pred_final, d):
    n = len(pred_final)
    cvc, src = d["class_voxel_counts"], d["pred_class"]
    return cvc[np.arange(n), pred_final] - cvc[np.arange(n), src]


def patient_utility(du, d):
    df = pd.DataFrame({"patient": d["patients"], "du": du, "size": d["size_voxels"]})
    g = df.groupby("patient").agg(s=("du", "sum"), sz=("size", "sum"))
    return g["s"] / g["sz"]


def boot_ci(vals, n_boot=N_BOOT, seed=BOOT_SEED):
    vals = np.asarray(vals, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(vals), size=(n_boot, len(vals)))
    means = vals[idx].mean(axis=1)
    return float(vals.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def edit_counts(pred_final, d):
    du = delta_u(pred_final, d)
    acted = pred_final != d["pred_class"]
    return {
        "acted": int(acted.sum()),
        "beneficial": int((acted & (du > 0)).sum()),
        "harmful": int((acted & (du < 0)).sum()),
        "neutral": int((acted & (du == 0)).sum()),
    }


def class_edit_counts(pred_final, d, target):
    du = delta_u(pred_final, d)
    m = (pred_final != d["pred_class"]) & (pred_final == target)
    return {"n": int(m.sum()), "beneficial": int((m & (du > 0)).sum()),
            "harmful": int((m & (du < 0)).sum()), "net": int((m & (du > 0)).sum() - (m & (du < 0)).sum())}


# ---------------------------------------------------------------- decision rules from p_final
def action_score_and_target(p_final, src):
    """score = P(change) = 1 - P(final=source); target = argmax over non-source classes."""
    n = len(src)
    masked = p_final.copy()
    masked[np.arange(n), src] = -1.0
    target = masked.argmax(axis=1)
    score = 1.0 - p_final[np.arange(n), src]
    return score, target


def coverage_curve(p_final, d, ks, order_score=None):
    """Act on the global top-K regions by score; returns per-K patient-level utility vectors."""
    src = d["pred_class"]
    score, target = action_score_and_target(p_final, src)
    if order_score is not None:
        score = order_score
    order = np.argsort(-score, kind="stable")
    out = {}
    du_target = delta_u(target, d)
    for k in ks:
        sel = np.zeros(len(src), bool)
        sel[order[:k]] = True
        du = np.where(sel, du_target, 0.0)
        out[k] = patient_utility(du, d)
    return out, order, target


# ---------------------------------------------------------------- generalized trainers
def _split_patients(train_patients, seed, tag):
    rng = np.random.default_rng(1000 * seed + tag)
    sh = rng.permutation(np.array(sorted(train_patients)))
    n_val = max(1, int(round(VAL_FRAC * len(sh))))
    return set(sh[n_val:].tolist()), set(sh[:n_val].tolist())


def _wnll_T(model, x, src, y, w):
    log_p_act, log_p_keep, log_p_trans, _ = model.forward(x, src)
    is_keep = y == src
    lp_tr = log_p_trans.gather(1, y.unsqueeze(1)).squeeze(1)
    lp = torch.where(is_keep, log_p_keep, log_p_act + lp_tr)
    return -(lp * w).sum() / w.sum()


def _wnll_M(model, x, y, w):
    lp = torch.log_softmax(model(x), dim=1).gather(1, y.unsqueeze(1)).squeeze(1)
    return -(lp * w).sum() / w.sum()


def _train(model, loss_fn, tr, va, torch_seed):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    best, best_state, bad = np.inf, None, 0
    for ep in range(MAX_EPOCHS):
        model.train(); opt.zero_grad(); loss_fn(*tr).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            v = loss_fn(*va).item()
        if v < best - 1e-5:
            best, best_state, bad = v, {k: t.clone() for k, t in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(best_state); model.eval()
    return model


def fit_predict(arch, X, y_final, src, patients, train_mask, test_mask, seed, tag, weight=None, X_extra=None):
    """Train one head on train_mask rows (patient-grouped internal fit/val split, scaler on all
    train rows), return P_final on test_mask rows (and on X_extra rows if given).
    arch in {"T","M"}; weight = per-row loss weight (None => unweighted, as in the original study)."""
    tr_pat = set(patients[train_mask])
    fit_p, val_p = _split_patients(tr_pat, seed, tag)
    fit = train_mask & np.isin(patients, list(fit_p))
    val = train_mask & np.isin(patients, list(val_p))
    sc = StandardScaler().fit(X[train_mask])
    t = lambda a, dt=torch.float32: torch.tensor(a, dtype=dt)
    has_test = test_mask is not None and test_mask.any()
    Xf, Xv = t(sc.transform(X[fit])), t(sc.transform(X[val]))
    Xt = t(sc.transform(X[test_mask])) if has_test else None
    yf, yv = t(y_final[fit], torch.long), t(y_final[val], torch.long)
    sf, sv = t(src[fit], torch.long), t(src[val], torch.long)
    st = t(src[test_mask], torch.long) if has_test else None
    wf = t(weight[fit]) if weight is not None else torch.ones(int(fit.sum()))
    wv = t(weight[val]) if weight is not None else torch.ones(int(val.sum()))
    if weight is not None:  # normalise to mean 1 on the fit rows for a comparable learning-rate scale
        m = wf.mean(); wf, wv = wf / m, wv / m
    torch.manual_seed(seed * 10 + tag)
    d_in = X.shape[1]
    if arch == "T":
        model = ModelT(d_in)
        model = _train(model, lambda x, s, y, w: _wnll_T(model, x, s, y, w), (Xf, sf, yf, wf), (Xv, sv, yv, wv), seed)
        res = {}
        if has_test:
            pf, pa, _ = model.predict_p_final(Xt, st)
            res = {"p_final": pf.numpy(), "p_act": pa.numpy()}
        if X_extra is not None:
            xe, se = X_extra
            pfe, pae, _ = model.predict_p_final(t(sc.transform(xe)), t(se, torch.long))
            res["extra_p_final"], res["extra_p_act"] = pfe.numpy(), pae.numpy()
    else:
        model = ModelM(d_in)
        model = _train(model, lambda x, y, w: _wnll_M(model, x, y, w), (Xf, yf, wf), (Xv, yv, wv), seed)
        res = {"p_final": model.predict_p_final(Xt).numpy()} if has_test else {}
        if X_extra is not None:
            xe, se = X_extra
            res["extra_p_final"] = model.predict_p_final(t(sc.transform(xe))).numpy()
    return res


# ---------------------------------------------------------------- protocols
N_INNER = 5


def p2_inner_assignment(d):
    """Per-network nested protocol: inside base network f (= frozen fold f), its patients are split
    into 5 patient-grouped inner folds (deterministic). Corrector f is trained/tested ONLY on
    regions whose features come from network f, so no corrector patient was ever in network f's
    base-model training set (that set is the other four folds)."""
    inner = np.zeros(d["n"], dtype=int)
    assign = {}
    for f in range(5):
        pats = np.array(sorted(set(d["patients"][d["fold"] == f])))
        perm = np.random.default_rng(777 + f).permutation(len(pats))
        for rank, i in enumerate(perm):
            assign[pats[i]] = {"network": f, "inner_fold": int(rank % N_INNER)}
    for i, p in enumerate(d["patients"]):
        inner[i] = assign[p]["inner_fold"]
    return inner, assign


def iter_splits(d, protocol):
    """Yield (tag, train_mask, test_mask)."""
    fold = d["fold"]
    if protocol in ("P0", "P1"):
        for k in range(5):
            yield k, fold != k, fold == k
    elif protocol == "P2":
        inner, _ = p2_inner_assignment(d)
        for f in range(5):
            for j in range(N_INNER):
                yield f * 5 + j, (fold == f) & (inner != j), (fold == f) & (inner == j)
    else:
        raise ValueError(protocol)


PROTOCOL_FEATURES = {"P0": "S2", "P1": "S1", "P2": "S2"}


def run_protocol(d, protocol, arch, seed, weight=None):
    """Out-of-fold P_final (n,5) (and p_act for T) under a protocol, for one arch/seed."""
    X = features(d, PROTOCOL_FEATURES[protocol])
    n = d["n"]
    pf, pa = np.zeros((n, 5)), np.zeros(n)
    for tag, tr, te in iter_splits(d, protocol):
        r = fit_predict(arch, X, d["y_final"], d["pred_class"], d["patients"], tr, te, seed, tag, weight)
        pf[te] = r["p_final"]
        if arch == "T":
            pa[te] = r["p_act"]
    return pf, pa


class ActOnly(torch.nn.Module):
    """Trunk + single sigmoid action head, trained alone with BCE on label_act (R1.5 ablation:
    same trunk, features, optimiser, early-stopping rule as Model T; no transition head)."""

    def __init__(self, d_in):
        super().__init__()
        from k6_models import make_trunk, HIDDEN2
        self.trunk = make_trunk(d_in)
        self.head = torch.nn.Linear(HIDDEN2, 1)

    def forward(self, x):
        return self.head(self.trunk(x)).squeeze(-1)


def fit_predict_act_only(X, label_act, patients, train_mask, test_mask, seed, tag):
    tr_pat = set(patients[train_mask])
    fit_p, val_p = _split_patients(tr_pat, seed, tag)
    fit = train_mask & np.isin(patients, list(fit_p))
    val = train_mask & np.isin(patients, list(val_p))
    sc = StandardScaler().fit(X[train_mask])
    t = lambda a, dt=torch.float32: torch.tensor(a, dtype=dt)
    Xf, Xv, Xt = t(sc.transform(X[fit])), t(sc.transform(X[val])), t(sc.transform(X[test_mask]))
    yf, yv = t(label_act[fit]), t(label_act[val])
    torch.manual_seed(seed * 10 + tag)
    model = ActOnly(X.shape[1])
    bce = torch.nn.functional.binary_cross_entropy_with_logits
    model = _train(model, lambda x, y: bce(model(x), y), (Xf, yf), (Xv, yv), seed)
    with torch.no_grad():
        return torch.sigmoid(model(Xt)).numpy()
