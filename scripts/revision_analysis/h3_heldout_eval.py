"""ONE-SHOT evaluation on the 50 reserved patients (140 cases).  See REVISION_PROTOCOL_LOCK.md section 4.
Inference only on the frozen base networks; correctors are trained on DEVELOPMENT data only.

Replicate structure: each of the five base networks scores all reserved cases.  A patient's value is the
mean over the five replicates.  P2: corrector f = trained on all development regions of network f.
P1 / P0: correctors trained on all 439 development cases, applied to regions from each network.
"""
import json
import sys
from multiprocessing import Pool

import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1])); from mriseg_paths import P as _P, CODE as _CODE  # noqa: E402
sys.path.insert(0, str(_CODE / "revision_analysis"))
from rev_common import *  # noqa
import a6_dice_hd95 as A6
from dice_summary import summarize
from sklearn.metrics import roc_auc_score


def main():
    HELD = OUT / "heldout"
    d = load_dev()
    recs = {f: json.loads((HELD / f"fold{f}" / "regions.json").read_text()) for f in range(5)}


    def region_arrays(rs):
        S0 = np.array([r["mean_probs"] for r in rs]); pc = np.array([r["pred_class"] for r in rs])
        oh = np.zeros((len(rs), 5)); oh[np.arange(len(rs)), pc] = 1
        size = np.array([r["size_voxels_mm3"] for r in rs], float)
        S1 = np.concatenate([S0, oh, np.log1p(size)[:, None]], 1)
        S2 = np.concatenate([S1, np.array([r["decoder_feat_mean"] for r in rs]), np.array([r["decoder_feat_std"] for r in rs])], 1)
        cvc = np.array([r["class_voxel_counts"] for r in rs], float)
        mx = cvc.max(1, keepdims=True)
        # oracle label with the same keep-biased tie-break as the development analysis
        tied = cvc == mx
        a_star = np.where(tied[np.arange(len(rs)), pc], pc, tied.argmax(1))
        return {"n": len(rs), "patients": np.array([r["patient"] for r in rs]), "cases": np.array([r["case"] for r in rs]),
                "pred_class": pc, "size_voxels": size, "class_voxel_counts": cvc, "S1": S1, "S2": S2,
                "label_act": (a_star != pc).astype(int)}


    H = {f: region_arrays(recs[f]) for f in range(5)}
    print("reserved regions per network:", {f: H[f]["n"] for f in H}, flush=True)

    # ----------------------------------------------------------------- train on development data, predict reserved
    X_dev = {"S1": features(d, "S1"), "S2": features(d, "S2")}
    allm = np.ones(d["n"], bool)
    P = {}      # P[(protocol, arch)][f] -> pooled p_final on reserved regions of network f
    for protocol in ("P2", "P1", "P0"):
        fs = "S1" if protocol == "P1" else "S2"
        for arch in ("T", "M"):
            P[(protocol, arch)] = {f: [] for f in range(5)}
            for seed in SEEDS:
                if protocol == "P2":
                    for f in range(5):
                        r = fit_predict(arch, X_dev[fs], d["y_final"], d["pred_class"], d["patients"], d["fold"] == f, None, seed, 100 + f,
                                        X_extra=(H[f][fs], H[f]["pred_class"]))
                        P[(protocol, arch)][f].append(r["extra_p_final"])
                else:
                    for f in range(5):
                        r = fit_predict(arch, X_dev[fs], d["y_final"], d["pred_class"], d["patients"], allm, None, seed, 200,
                                        X_extra=(H[f][fs], H[f]["pred_class"]))
                        P[(protocol, arch)][f].append(r["extra_p_final"])
            print("trained", protocol, arch, flush=True)
    pooled = {k: {f: np.mean(v[f], 0) for f in range(5)} for k, v in P.items()}

    # ----------------------------------------------------------------- utility endpoints
    pats = sorted(set(H[0]["patients"]))
    results = {"n_patients": len(pats), "n_cases": int(len(set(H[0]["cases"]))), "n_regions_per_network": {f: H[f]["n"] for f in H}}
    decisions = {f: {"src": H[f]["pred_class"].copy()} for f in range(5)}
    for f in range(5):
        cvc = H[f]["class_voxel_counts"]; pc = H[f]["pred_class"]
        gain = cvc - cvc[np.arange(H[f]["n"]), pc][:, None]
        decisions[f]["oracle_target"] = np.where(gain.max(1) > 0, gain.argmax(1), pc)

    for protocol in ("P2", "P1", "P0"):
        pu = {"T": [], "M": [], "M_matchedK": []}; net_res = {}
        cnt = {"T": [], "M": []}
        for f in range(5):
            dd = H[f]
            pT, pM = pooled[(protocol, "T")][f], pooled[(protocol, "M")][f]
            predT, predM = pT.argmax(1), pM.argmax(1)
            N_T = int((predT != dd["pred_class"]).sum())
            scM, tgM = action_score_and_target(pM, dd["pred_class"])
            sel = np.zeros(dd["n"], bool); sel[np.argsort(-scM, kind="stable")[:N_T]] = True
            predMK = np.where(sel, tgM, dd["pred_class"])
            uT, uM, uMK = (patient_utility(delta_u(p, dd), dd).reindex(pats) for p in (predT, predM, predMK))
            pu["T"].append(uT); pu["M"].append(uM); pu["M_matchedK"].append(uMK)
            cnt["T"].append(edit_counts(predT, dd) | {"rc": class_edit_counts(predT, dd, 4), "netc": class_edit_counts(predT, dd, 1)})
            cnt["M"].append(edit_counts(predM, dd) | {"rc": class_edit_counts(predM, dd, 4), "netc": class_edit_counts(predM, dd, 1)})
            net_res[str(f)] = {"T_mean": float(uT.mean()), "M_mean": float(uM.mean()), "diff": list(boot_ci((uT - uM).dropna().values)),
                               "T_act_auroc": float(roc_auc_score(dd["label_act"], pT.sum(1) * 0 + 1 - pT[np.arange(dd["n"]), dd["pred_class"]])) if len(set(dd["label_act"])) == 2 else None}
            decisions[f][f"{protocol}_unw__T"] = predT; decisions[f][f"{protocol}_unw__M"] = predM
            decisions[f][f"{protocol}_unw__M_matchedK"] = predMK
        mT = pd.concat(pu["T"], axis=1).mean(axis=1); mM = pd.concat(pu["M"], axis=1).mean(axis=1)
        mMK = pd.concat(pu["M_matchedK"], axis=1).mean(axis=1)
        ok = mT.notna() & mM.notna() & mMK.notna()          # patients with no candidate region in any replicate are undefined
        results.setdefault("n_patients_defined", {})[protocol] = int(ok.sum())
        mT, mM, mMK = mT[ok], mM[ok], mMK[ok]
        results[protocol] = {
            "T_mean_util": float(mT.mean()), "M_mean_util": float(mM.mean()),
            "T_mean_util_ci": list(boot_ci(mT.values)), "M_mean_util_ci": list(boot_ci(mM.values)),
            "T_vs_keep_patients_positive": int((mT > 0).sum()),
            "paired_diff_T_minus_M": list(boot_ci((mT - mM).values)), "T_higher_patients": int((mT > mM).sum()),
            "paired_diff_T_minus_M_at_typed_coverage": list(boot_ci((mT - mMK).values)),
            "mean_util_M_at_typed_coverage": float(mMK.mean()),
            "per_network": net_res,
            "patient_values": {"patients": [str(p) for p in mT.index], "T": [float(x) for x in mT.values],
                               "M": [float(x) for x in mM.values], "M_matched": [float(x) for x in mMK.values]},
            "counts_mean_over_networks": {a: {k: (float(np.mean([c[k] for c in cnt[a]])) if not isinstance(cnt[a][0][k], dict) else
                                               {kk: float(np.mean([c[k][kk] for c in cnt[a]])) for kk in cnt[a][0][k]})
                                              for k in cnt[a][0]} for a in cnt},
        }
        r = results[protocol]
        print(f"{protocol}: T {r['T_mean_util']:+.4f} M {r['M_mean_util']:+.4f} diff {r['paired_diff_T_minus_M'][0]:+.4f} "
              f"[{r['paired_diff_T_minus_M'][1]:+.4f},{r['paired_diff_T_minus_M'][2]:+.4f}]", flush=True)
    (OUT / "H_RESULTS_UTILITY.json").write_text(json.dumps(results, indent=1, default=float))

    # ----------------------------------------------------------------- Dice / HD95 on the reserved cohort
    names = {"baseline": "src", "P0_typed": "P0_unw__T", "P0_mono": "P0_unw__M", "P2_typed": "P2_unw__T",
             "P2_mono": "P2_unw__M", "P2_mono_matchedK": "P2_unw__M_matchedK", "oracle_relabel": "oracle_target"}
    jobs = []
    for f in range(5):
        by_case = {}
        for i, r in enumerate(recs[f]):
            by_case.setdefault(r["case"], []).append((i, r))
        for case, items in by_case.items():
            idx = [i for i, _ in items]
            jobs.append((f"{case}", HELD / f"fold{f}" / "masks" / f"{case}.npz", [r for _, r in items],
                         {v: [int(decisions[f][k][i]) for i in idx] for v, k in names.items()}, f))
    for f in range(5):                      # cases with no candidate region are unchanged by any correction but belong to the baseline
        have = {j[0] for j in jobs if j[4] == f}
        for p_ in sorted((HELD / f"fold{f}" / "masks").glob("*.npz")):
            if p_.stem not in have:
                jobs.append((p_.stem, p_, [], {v: [] for v in names}, f))
    with Pool(10) as p:
        res = p.map(A6.work, [j[:4] for j in jobs], chunksize=4)
    rows = []
    for j, rr in zip(jobs, res):
        for r in rr:
            r["network"] = j[4]; rows.append(r)
    df = pd.DataFrame(rows); df["patient"] = df["case"].str.extract(r"(PatientID_\d+)")[0]
    df.to_csv(OUT / "H_DICE_HD95_heldout.csv", index=False)
    summ = summarize(df, list(names))
    (OUT / "H_RESULTS_DICE_HD95.json").write_text(json.dumps(summ, indent=1, default=float))
    print("DONE heldout")



if __name__ == "__main__":
    main()
