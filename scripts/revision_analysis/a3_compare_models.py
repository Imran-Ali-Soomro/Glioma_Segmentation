"""R1.1 / R2.4 (matched coverage), R1.7 (objective alignment), R2.8 (variability):
typed vs monolithic under P0 / P1 / P2, unweighted and size-weighted, with matched-coverage
analysis and a component-swap control. Development cohort only."""
import json
import sys

import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1])); from mriseg_paths import P as _P, CODE as _CODE  # noqa: E402
sys.path.insert(0, str(_CODE / "revision_analysis"))
from rev_common import *  # noqa
from sklearn.metrics import roc_auc_score

d = load_dev()
y, src, pat, fold = d["label_act"], d["pred_class"], d["patients"], d["fold"]
n = d["n"]


def load_oof(name):
    if name == "P0_unw" and VARIANT != "slide":
        z = np.load(REPO / "PTDR_K6_OOF_PREDICTIONS.npz")      # original saved predictions, verbatim (tiled pipeline)
    else:
        z = np.load(OUT / f"oof_{name.replace('_unw','').replace('_size','')}_{name.split('_')[1]}.npz")
    out = {a: [z[f"{a}_p_final_seed{s}"] for s in SEEDS] for a in ("T", "M")}
    out["T_act"] = [z[f"T_p_act_seed{s}"] for s in SEEDS]
    return out


def pooled(lst):
    return np.mean(lst, axis=0)


results = {}
curves = {}
decisions = {"src": src.copy(), "oracle_target": np.where(d["label_act"] == 1, d["transition_target"], src)}
KS = [25, 50, 100, 200, 400, 600, 800, 1000, 1250, 1500, 2000, 2500, 3000, 3500, 4000, n]

for name in ["P0_unw", "P1_unw", "P2_unw", "P1_size", "P2_size"]:
    o = load_oof(name)
    pT, pM = pooled(o["T"]), pooled(o["M"])
    predT, predM = pT.argmax(1), pM.argmax(1)
    uT, uM = patient_utility(delta_u(predT, d), d), patient_utility(delta_u(predM, d), d)
    diff = (uT - uM)
    m, lo, hi = boot_ci(diff.values)
    r = {
        "T_mean_util": float(uT.mean()), "M_mean_util": float(uM.mean()),
        "paired_diff_T_minus_M": [m, lo, hi], "T_higher_patients": int((diff > 0).sum()),
        "n_patients": int(len(diff)),
        "T_counts": edit_counts(predT, d), "M_counts": edit_counts(predM, d),
        "T_rc": class_edit_counts(predT, d, 4), "M_rc": class_edit_counts(predM, d, 4),
        "T_netc": class_edit_counts(predT, d, 1), "M_netc": class_edit_counts(predM, d, 1),
        "T_act_auroc_pooled": float(roc_auc_score(y, pooled(o["T_act"]))),
        "M_implied_act_auroc_pooled": float(roc_auc_score(y, 1 - pM[np.arange(n), src])),
        "per_seed": [], "per_network": {},
    }
    for s in SEEDS:
        pdiff = patient_utility(delta_u(o["T"][s].argmax(1), d), d) - patient_utility(delta_u(o["M"][s].argmax(1), d), d)
        r["per_seed"].append({"seed": s, "T_mean": float(patient_utility(delta_u(o["T"][s].argmax(1), d), d).mean()),
                              "M_mean": float(patient_utility(delta_u(o["M"][s].argmax(1), d), d).mean()),
                              "diff": list(boot_ci(pdiff.values))})
    for f in range(5):                                       # variability across base networks
        pf = sorted(set(pat[fold == f]))
        dd = diff.loc[pf].values
        r["per_network"][str(f)] = {"n_patients": len(pf), "diff": list(boot_ci(dd)),
                                    "T_mean": float(uT.loc[pf].mean()), "M_mean": float(uM.loc[pf].mean())}
    # ---------------- matched coverage ----------------
    scT, tgT = action_score_and_target(pT, src)
    scM, tgM = action_score_and_target(pM, src)
    N_T, N_M = r["T_counts"]["acted"], r["M_counts"]["acted"]
    # fixed-score controls at K = typed default coverage
    def util_at(score, target, K):
        order = np.argsort(-score, kind="stable")
        sel = np.zeros(n, bool); sel[order[:K]] = True
        return patient_utility(np.where(sel, delta_u(target, d), 0.0), d)
    mc = {}
    for K, tag in ((N_T, "K=typed_default_coverage"), (N_M, "K=mono_default_coverage")):
        uTK, uMK = util_at(scT, tgT, K), util_at(scM, tgM, K)
        dK = uTK - uMK
        mc[tag] = {"K": int(K), "T_mean": float(uTK.mean()), "M_mean": float(uMK.mean()),
                   "paired_diff_T_minus_M": list(boot_ci(dK.values)), "T_higher_patients": int((dK > 0).sum()),
                   "T_score__M_target_mean": float(util_at(scT, tgM, K).mean()),
                   "M_score__T_target_mean": float(util_at(scM, tgT, K).mean()),
                   "T_score__T_target_mean": float(uTK.mean()), "M_score__M_target_mean": float(uMK.mean())}
        # confirm what each swap isolates: authorisation (score) vs target selection
        a = util_at(scT, tgT, K) - util_at(scM, tgT, K)      # same target, different authorisation ranking
        b = util_at(scT, tgT, K) - util_at(scT, tgM, K)      # same ranking, different target
        mc[tag]["authorisation_effect_same_target(T_score - M_score)"] = list(boot_ci(a.values))
        mc[tag]["target_effect_same_ranking(T_target - M_target)"] = list(boot_ci(b.values))
    r["matched_coverage"] = mc
    # full curves
    cT, cM, cOr, cRn = [], [], [], []
    best_du = np.max(d["class_voxel_counts"] - d["class_voxel_counts"][np.arange(n), src][:, None], axis=1)
    or_target = (d["class_voxel_counts"] - d["class_voxel_counts"][np.arange(n), src][:, None]).argmax(1)
    rng = np.random.default_rng(5)
    for K in KS:
        cT.append(float(util_at(scT, tgT, K).mean())); cM.append(float(util_at(scM, tgM, K).mean()))
        cOr.append(float(util_at(best_du, or_target, K).mean()))
        cRn.append(float(np.mean([util_at(rng.random(n), tgM, K).mean() for _ in range(5)])))
    curves[name] = {"K": KS, "typed": cT, "mono": cM, "oracle_order_oracle_target": cOr,
                    "random_order_mono_target": cRn, "typed_default_K": int(N_T), "mono_default_K": int(N_M)}
    decisions[f"{name}__T"] = predT.copy(); decisions[f"{name}__M"] = predM.copy()
    selK = np.zeros(n, bool); selK[np.argsort(-scM, kind="stable")[:N_T]] = True
    decisions[f"{name}__M_matchedK"] = np.where(selK, tgM, src)
    results[name] = r
    print(f"{name}: T={uT.mean():+.4f} M={uM.mean():+.4f} diff={m:+.4f} [{lo:+.4f},{hi:+.4f}] "
          f"acted T={N_T} M={N_M} | matched-K(T) diff={mc['K=typed_default_coverage']['paired_diff_T_minus_M'][0]:+.4f} "
          f"[{mc['K=typed_default_coverage']['paired_diff_T_minus_M'][1]:+.4f},{mc['K=typed_default_coverage']['paired_diff_T_minus_M'][2]:+.4f}] "
          f"| T act AUROC {r['T_act_auroc_pooled']:.3f}", flush=True)

np.savez_compressed(OUT / "A3_VARIANT_DECISIONS.npz", **decisions)
(OUT / "A3_COMPARISON.json").write_text(json.dumps(results, indent=1))
(OUT / "A3_COVERAGE_CURVES.json").write_text(json.dumps(curves, indent=1))
print("DONE")
