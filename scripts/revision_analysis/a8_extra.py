"""Derived values used in the manuscript text (clean key names for render_manuscript placeholders)."""
import json
import sys

import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1])); from mriseg_paths import P as _P, CODE as _CODE  # noqa: E402
sys.path.insert(0, str(_CODE / "revision_analysis"))
from rev_common import *  # noqa

d = load_dev()
A3 = json.loads((OUT / "A3_COMPARISON.json").read_text())
C = json.loads((OUT / "A3_COVERAGE_CURVES.json").read_text())
A5 = json.loads((OUT / "A5_CEILING_DECOMPOSITION.json").read_text())
D = np.load(OUT / "A3_VARIANT_DECISIONS.npz")
out = {}

# ---- topology and ceiling
E = A5["recomputed_total_error_from_masks"]
act = d["label_act"] == 1
tt = d["transition_target"]
out["topo"] = {
    "E": E, "over_pct": A5["topology_pct"]["over_segmentation"], "missed_pct": A5["topology_pct"]["missed_tissue"],
    "wrong_pct": A5["topology_pct"]["wrong_fg_class"], "n_regions": int(d["n"]), "n_cases_with_regions": int(len(set(d["cases"]))),
    "n_cases_without_regions": 439 - int(len(set(d["cases"]))), "actionable_pct": 100 * float(act.mean()), "n_actionable": int(act.sum()),
    "bg_target_pct": 100 * float((tt[act] == 0).mean()),
    "netc_target_n": int((tt[act] == 1).sum()), "snfh_target_n": int((tt[act] == 2).sum()),
    "et_target_n": int((tt[act] == 3).sum()), "rc_target_n": int((tt[act] == 4).sum()), "bg_target_n": int((tt[act] == 0).sum()),
}
p = A5["pct_of_total_error"]; cc = A5["counterfactual_ceilings_pct"]; g = A5["gap_explained"]
out["ceil"] = {
    "ceiling_pct": cc["as_published (min size 8, heterogeneous regions)"], "noMin_pct": cc["no minimum size (heterogeneous regions)"],
    "homog_pct": cc["homogeneous regions, min size 8"], "all_nonmissed_pct": cc["no minimum size AND homogeneous (= all non-missed error)"],
    "heter_pct": p["heterogeneity inside regions >=8 (H)"], "sub8_recov_pct": p["sub-8 components, recoverable by oracle relabel"],
    "sub8_resid_pct": p["sub-8 components, still heterogeneous"], "gap_pct": g["gap_pct_of_error"],
    "heter_share_of_gap": 100 * g["heterogeneity_share_of_gap"], "sub8_share_of_gap": 100 * g["sub8_share_of_gap"], "n_lt8": A5["n_components_lt8"],
}


# ---- comparison summaries
def summ(key, ckey):
    r = A3[key]; lo, hi = r["paired_diff_T_minus_M"][1:]
    orc = C[ckey]["oracle_order_oracle_target"][-1]
    seeds = [s["diff"][0] for s in r["per_seed"]]; nets = [v["diff"] for v in r["per_network"].values()]
    return {
        "T_util": r["T_mean_util"], "M_util": r["M_mean_util"], "diff": r["paired_diff_T_minus_M"][0], "lo": lo, "hi": hi,
        "bound": max(abs(lo), abs(hi)), "oracle_ceiling": orc, "T_frac_oracle": 100 * r["T_mean_util"] / orc,
        "M_frac_oracle": 100 * r["M_mean_util"] / orc, "diff_frac_oracle": 100 * r["paired_diff_T_minus_M"][0] / orc,
        "bound_frac_oracle": 100 * max(abs(lo), abs(hi)) / orc,
        "seed_min": min(seeds), "seed_max": max(seeds), "net_min": min(n[0] for n in nets), "net_max": max(n[0] for n in nets),
        "n_net_pos": int(sum(n[0] > 0 for n in nets)), "n_net_excl0": int(sum((n[1] > 0 or n[2] < 0) for n in nets)),
        "n_seed_pos": int(sum(s > 0 for s in seeds)),
    }


out["p2"] = summ("P2_unw", "P2_unw"); out["p1"] = summ("P1_unw", "P1_unw"); out["p0"] = summ("P0_unw", "P0_unw")
out["p2size"] = summ("P2_size", "P2_size")

# ---- edits by target class (voxel utility)
names = ["BG", "NETC", "SNFH", "ET", "RC"]
src = d["pred_class"]
for arch in ("T", "M"):
    pred = D[f"P2_unw__{arch}"]; du = delta_u(pred, d); acted = pred != src
    tot = float(du.sum())
    out[f"edits_P2_{arch}"] = {n.lower(): {
        "n": int((acted & (pred == k)).sum()), "ben": int((acted & (pred == k) & (du > 0)).sum()),
        "harm": int((acted & (pred == k) & (du < 0)).sum()), "du_sum": float(du[acted & (pred == k)].sum()),
        "du_share_pct": 100 * float(du[acted & (pred == k)].sum()) / tot if tot else 0.0} for k, n in enumerate(names)}
    out[f"edits_P2_{arch}"]["total_du"] = tot
    out[f"edits_P2_{arch}"]["pos_du"] = float(du[du > 0].sum()); out[f"edits_P2_{arch}"]["neg_du"] = float(du[du < 0].sum())
(OUT / "A8_EXTRA.json").write_text(json.dumps(out, indent=1))
print(json.dumps({k: out[k] for k in ("topo", "p2", "p1")}, indent=1)[:3500])

# ---- reserved cohort derived values (if the one-shot evaluation exists for this variant)
hp = OUT / "H_RESULTS_UTILITY.json"
if hp.exists():
    HU = json.loads(hp.read_text())
    nets = [v["diff"] for v in HU["P2"]["per_network"].values()]
    counts = list(HU["n_regions_per_network"].values())
    out["res"] = {
        "net_min": min(n[0] for n in nets), "net_max": max(n[0] for n in nets),
        "n_net_excl0": int(sum((n[1] > 0 or n[2] < 0) for n in nets)),
        "regions_min": min(counts), "regions_max": max(counts), "n_defined": HU["n_patients_defined"]["P2"],
        "n_undefined": 50 - HU["n_patients_defined"]["P2"],
        "T_acted": HU["P2"]["counts_mean_over_networks"]["T"]["acted"], "M_acted": HU["P2"]["counts_mean_over_networks"]["M"]["acted"],
        "T_ben": HU["P2"]["counts_mean_over_networks"]["T"]["beneficial"], "T_harm": HU["P2"]["counts_mean_over_networks"]["T"]["harmful"],
    }
    (OUT / "A8_EXTRA.json").write_text(json.dumps(out, indent=1))
    print("reserved extras:", out["res"])
