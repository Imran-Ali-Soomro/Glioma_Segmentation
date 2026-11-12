"""R1.6 / R2.6: calibration and RC-mitigation experiments, on the corrected (P2, nested per-network)
protocol and, for contrast, on the original frozen-fold protocol (P0).

Pipeline replicated (the 'separately fitted diagnostic' of the original paper):
  Q_act = S2 MLP(16) (sklearn, early stopping), Q_transition = S1 logistic regression, unweighted,
  oracle-actionable rows only, source-masked to the target set seen in training for that source class,
  HARD_GATE controller (act if Q_act >= tau, then argmax masked Q_transition).
Mitigations tested: Platt scaling, isotonic regression (both cross-fitted: calibrator fit on the OTHER
folds' out-of-fold scores), global threshold sweep, RC-specific authorisation threshold.
"""
import json
import sys

import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1])); from mriseg_paths import P as _P, CODE as _CODE  # noqa: E402
sys.path.insert(0, str(_CODE / "revision_analysis"))
from rev_common import *  # noqa
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.neural_network import MLPClassifier

d = load_dev()
y, src, pat, fold, n = d["label_act"], d["pred_class"], d["patients"], d["fold"], d["n"]
S1, S2 = features(d, "S1"), features(d, "S2")
tgt = d["transition_target"]
cvc = d["class_voxel_counts"]
inner, _ = p2_inner_assignment(d)


def run_pipeline(protocol):
    p_act = np.zeros(n); p_tr = np.zeros((n, 5))
    for tag, tr, te in iter_splits(d, "P2" if protocol == "P2" else "P0"):
        sc2 = StandardScaler().fit(S2[tr])
        m = MLPClassifier((16,), max_iter=500, random_state=0, early_stopping=True).fit(sc2.transform(S2[tr]), y[tr])
        p_act[te] = m.predict_proba(sc2.transform(S2[te]))[:, 1]
        tra = tr & (y == 1)
        sc1 = StandardScaler().fit(S1[tra])
        lr = LogisticRegression(max_iter=2000).fit(sc1.transform(S1[tra]), tgt[tra])
        pr = lr.predict_proba(sc1.transform(S1[te]))
        full = np.zeros((te.sum(), 5))
        for j, c in enumerate(lr.classes_):
            full[:, int(c)] = pr[:, j]
        # source masking to the permitted target set (targets observed for that source in training)
        permitted = {int(s): set(tgt[tra & (src == s)].tolist()) for s in set(src[tra])}
        for i, s in enumerate(src[te]):
            keep = np.zeros(5, bool)
            for t_ in permitted.get(int(s), set()):
                keep[t_] = True
            keep[int(s)] = False
            full[i] = np.where(keep, full[i], 0.0)
            tot = full[i].sum()
            full[i] = full[i] / tot if tot > 0 else full[i]
        p_tr[te] = full
    return p_act, p_tr


def ece(conf, correct, bins=12):
    e = np.linspace(0, 1, bins + 1); tot = 0.0
    for i in range(bins):
        m = (conf > e[i]) & (conf <= e[i + 1]) if i else (conf >= e[0]) & (conf <= e[1])
        if m.any():
            tot += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(tot)


def calib_row(p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    top = np.where(p >= .5, p, 1 - p); corr = ((p >= .5).astype(int) == y).astype(float)
    return {"auroc": float(roc_auc_score(y, p)), "brier": float(brier_score_loss(y, p)),
            "nll": float(log_loss(y, p)), "ece_top_label": ece(top, corr)}


def crossfit_calibrate(p, kind, protocol):
    """Calibrator for each held-out fold fitted on the OTHER folds' out-of-fold scores."""
    out = np.zeros(n)
    for tag, tr, te in iter_splits(d, "P2" if protocol == "P2" else "P0"):
        # restrict calibrator training to the same base network for P2 (feature spaces differ per network)
        fit_mask = tr if protocol != "P2" else tr
        lp = np.log(np.clip(p, 1e-9, 1 - 1e-9) / np.clip(1 - p, 1e-9, 1 - 1e-9))
        if kind == "platt":
            m = LogisticRegression(C=1e6, max_iter=1000).fit(lp[fit_mask].reshape(-1, 1), y[fit_mask])
            out[te] = m.predict_proba(lp[te].reshape(-1, 1))[:, 1]
        else:
            m = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p[fit_mask], y[fit_mask])
            out[te] = m.predict(p[te])
    return out


def hard_gate(p_act_, p_tr, tau_global=0.5, tau_rc=None):
    tgt_pred = p_tr.argmax(1)
    thr = np.full(n, tau_global)
    if tau_rc is not None:
        thr = np.where(tgt_pred == 4, tau_rc, thr)
    return np.where(p_act_ >= thr, tgt_pred, src)


def summarise(pred):
    du = delta_u(pred, d)
    pu = patient_utility(du, d)
    rc = class_edit_counts(pred, d, 4)
    acted_rc = (pred != src) & (pred == 4)
    return {"mean_util": float(pu.mean()), "net_voxel_utility": float(du.sum()), **edit_counts(pred, d),
            "rc_beneficial": rc["beneficial"], "rc_harmful": rc["harmful"], "rc_net": rc["net"],
            "rc_harm_false_authorisation_B": int((acted_rc & (du < 0) & (y == 0)).sum()),
            "rc_harm_wrong_target_D": int((acted_rc & (du < 0) & (y == 1)).sum())}


out = {}
for protocol in ("P0", "P2"):
    p_act, p_tr = run_pipeline(protocol)
    block = {"calibration": {"raw": calib_row(p_act)}}
    cal = {"raw": p_act}
    for kind in ("platt", "isotonic"):
        cal[kind] = crossfit_calibrate(p_act, kind, protocol)
        block["calibration"][kind] = calib_row(cal[kind])
    # source-conditioned calibration
    block["calibration_by_predicted_source_class"] = {
        CLASS_NAMES[c]: {"n": int((src == c).sum()), "brier": float(brier_score_loss(y[src == c], np.clip(p_act[src == c], 1e-9, 1 - 1e-9))),
                         "auroc": float(roc_auc_score(y[src == c], p_act[src == c])) if len(set(y[src == c])) == 2 else None}
        for c in range(1, 5)}
    # global-threshold sweep, per calibrator
    sweep = {}
    for kname, pa in cal.items():
        sweep[kname] = {f"{t:.2f}": summarise(hard_gate(pa, p_tr, t)) for t in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99)}
    block["global_threshold_sweep"] = sweep
    # RC-specific threshold: sweep, and a cross-fitted selection (choose tau_rc on the other folds)
    rc_sweep = {f"{t:.2f}": summarise(hard_gate(cal["raw"], p_tr, 0.5, t)) for t in (0.5, 0.7, 0.9, 0.95, 0.99, 0.999, 1.01)}
    block["rc_specific_threshold_sweep_raw"] = rc_sweep
    grid = [0.5, 0.7, 0.9, 0.95, 0.99, 0.999, 1.01]
    sel_pred = np.zeros(n, int); chosen = {}
    for tag, tr, te in iter_splits(d, "P2" if protocol == "P2" else "P0"):
        best_t, best_u = None, -np.inf
        for t in grid:
            pr = hard_gate(cal["raw"], p_tr, 0.5, t)
            u = delta_u(pr, d)[tr].sum()
            if u > best_u:
                best_u, best_t = u, t
        chosen[int(tag)] = best_t
        sel_pred[te] = hard_gate(cal["raw"], p_tr, 0.5, best_t)[te]
    block["rc_threshold_crossfitted_selection"] = {"summary": summarise(sel_pred), "chosen_tau_rc_per_split": chosen}
    block["keep_all_baseline"] = summarise(src.copy())
    out[protocol] = block
    c = block["calibration"]
    print(f"{protocol}: raw AUROC {c['raw']['auroc']:.3f} Brier {c['raw']['brier']:.3f} ECE {c['raw']['ece_top_label']:.3f} | "
          f"platt Brier {c['platt']['brier']:.3f} | isotonic Brier {c['isotonic']['brier']:.3f}")
    s5 = sweep["raw"]["0.50"]
    print(f"   hard-gate@0.5 RC: ben {s5['rc_beneficial']} harm {s5['rc_harmful']} net {s5['rc_net']} (B={s5['rc_harm_false_authorisation_B']}, D={s5['rc_harm_wrong_target_D']}) mean util {s5['mean_util']:+.4f}", flush=True)
    sx = block["rc_threshold_crossfitted_selection"]["summary"]
    print(f"   RC-specific xfit: RC ben {sx['rc_beneficial']} harm {sx['rc_harmful']} net {sx['rc_net']} mean util {sx['mean_util']:+.4f}", flush=True)

(OUT / "A4_CALIBRATION_RC.json").write_text(json.dumps(out, indent=1))
print("DONE")
