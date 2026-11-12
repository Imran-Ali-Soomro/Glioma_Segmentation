"""PTDR K6, sections 9, 11-14: primary utility evaluation, correction-quality
metrics, RC/NETC mandatory strata, patient-level paired bootstrap. Pure
post-hoc analysis of the frozen OOF predictions from k6_train_and_evaluate.py
-- no further training.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1])); from mriseg_paths import P as _P, CODE as _CODE  # noqa: E402
REPO_ROOT = _P(r"")
sys.path.insert(0, str(REPO_ROOT / "analysis" / "ptdr_stage1"))
from k6_data import load_k6_data, CLASS_NAMES  # noqa: E402

SEEDS = [0, 1, 2]
RNG_SEED = 20261002
N_BOOT = 2000


def region_level_metrics(predicted_final, data, run_label):
    source = data["pred_class"]
    cvc = data["class_voxel_counts"]
    label_act = data["label_act"]
    oracle_target = data["transition_target"]
    patients = data["patients"]
    n = len(predicted_final)

    delta_u = cvc[np.arange(n), predicted_final] - cvc[np.arange(n), source]
    acted = predicted_final != source
    beneficial = acted & (delta_u > 0)
    harmful = acted & (delta_u < 0)
    neutral = acted & (delta_u == 0)
    missed_action = (~acted) & (label_act == 1)

    n_acted = int(acted.sum())
    overall = {
        "run": run_label, "n_regions": n, "n_acted": n_acted,
        "beneficial_edits": int(beneficial.sum()), "harmful_edits": int(harmful.sum()),
        "neutral_edits": int(neutral.sum()), "missed_action_count": int(missed_action.sum()),
        "net_utility_total": float(delta_u.sum()),
        "total_positive_utility": float(delta_u[delta_u > 0].sum()),
        "total_negative_utility": float(delta_u[delta_u < 0].sum()),
        "beneficial_to_harmful_ratio": (int(beneficial.sum()) / int(harmful.sum())) if harmful.sum() > 0 else (float("inf") if beneficial.sum() > 0 else None),
        "harmful_edit_rate": (int(harmful.sum()) / n_acted) if n_acted else None,
        "correction_precision": (int(beneficial.sum()) / n_acted) if n_acted else None,
        "patient_support_acted": int(len(set(patients[acted]))),
    }

    def by_group(idx_mask, group_vals, as_target):
        out = {}
        sub_source, sub_pred, sub_delta = source[idx_mask], predicted_final[idx_mask], delta_u[idx_mask]
        sub_patients = patients[idx_mask]
        sub_ben, sub_harm = beneficial[idx_mask], harmful[idx_mask]
        for g in sorted(set(group_vals[idx_mask])):
            m = group_vals[idx_mask] == g
            nm = int(m.sum())
            if nm == 0:
                continue
            ben, harm = int(sub_ben[m].sum()), int(sub_harm[m].sum())
            entry = {
                "n": nm, "beneficial": ben, "harmful": harm, "net_corrected": ben - harm,
                "beneficial_to_harmful_ratio": (ben / harm) if harm > 0 else (float("inf") if ben > 0 else None),
                "harmful_edit_rate": harm / nm, "correction_precision": ben / nm,
                "patient_support_beneficial": int(len(set(sub_patients[m][sub_ben[m]]))) if ben else 0,
            }
            if as_target:
                true_total = int(np.sum((label_act == 1) & (oracle_target == g)))
                entry["true_oracle_actionable_total_for_target"] = true_total
                entry["correction_coverage_of_target_actionable"] = (ben / true_total) if true_total else None
            out[CLASS_NAMES[g]] = entry
        return out

    overall["by_source_class"] = by_group(acted, source, as_target=False)
    overall["by_predicted_final_target"] = by_group(acted, predicted_final, as_target=True)
    # by oracle target, restricted to genuinely oracle-actionable regions (oracle_target>=0)
    oracle_act_mask = acted & (oracle_target >= 0)
    overall["by_oracle_target_genuinely_actionable_subset"] = by_group(oracle_act_mask, oracle_target, as_target=True)

    # ---- mandatory NETC-source -> RC dedicated row ----
    netc_to_rc = acted & (source == 1) & (predicted_final == 4)
    n_ntr = int(netc_to_rc.sum())
    ben_ntr, harm_ntr = int(beneficial[netc_to_rc].sum()), int(harmful[netc_to_rc].sum())
    overall["netc_source_to_rc_target_dedicated_row"] = {
        "n": n_ntr, "beneficial": ben_ntr, "harmful": harm_ntr, "net_corrected": ben_ntr - harm_ntr,
        "beneficial_to_harmful_ratio": (ben_ntr / harm_ntr) if harm_ntr > 0 else (float("inf") if ben_ntr > 0 else None),
        "harmful_edit_rate": (harm_ntr / n_ntr) if n_ntr else None,
        "patient_support": int(len(set(patients[netc_to_rc]))) if n_ntr else 0,
    }
    return overall, delta_u


def patient_level_normalized_utility(delta_u, data):
    patients = data["patients"]
    sizes = data["size_voxels"]
    df = pd.DataFrame({"patient": patients, "delta_u": delta_u, "size": sizes})
    g = df.groupby("patient").agg(net_utility_sum=("delta_u", "sum"), total_region_voxels=("size", "sum"),
                                   n_candidate_regions=("delta_u", "size"))
    g["normalized_net_utility"] = g["net_utility_sum"] / g["total_region_voxels"]
    return g


def paired_bootstrap(diff_by_patient, n_boot=N_BOOT, seed=RNG_SEED):
    vals = diff_by_patient.values
    patients = diff_by_patient.index.values
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(vals), size=len(vals))
        means.append(vals[idx].mean())
    means = np.array(means)
    return {
        "observed_mean_diff_T_minus_M": float(vals.mean()),
        "bootstrap_ci_2.5": float(np.percentile(means, 2.5)),
        "bootstrap_ci_97.5": float(np.percentile(means, 97.5)),
        "excludes_zero": bool(np.percentile(means, 2.5) > 0 or np.percentile(means, 97.5) < 0),
        "favors": "T" if vals.mean() > 0 else ("M" if vals.mean() < 0 else "tie"),
        "n_patients": len(vals),
    }


def main():
    data = load_k6_data()
    npz = np.load(REPO_ROOT / "PTDR_K6_OOF_PREDICTIONS.npz")

    seed_results = {}
    patient_rows = []
    delta_u_store = {}  # (model, run) -> delta_u array

    for model in ["T", "M"]:
        p_finals = [npz[f"{model}_p_final_seed{s}"] for s in SEEDS]
        pooled_p_final = np.mean(p_finals, axis=0)

        for s, p_final in zip(SEEDS, p_finals):
            predicted = p_final.argmax(axis=1)
            metrics, delta_u = region_level_metrics(predicted, data, f"{model}_seed{s}")
            seed_results.setdefault(model, {})[f"seed{s}"] = metrics
            delta_u_store[(model, f"seed{s}")] = delta_u
            pu = patient_level_normalized_utility(delta_u, data)
            for patient, row in pu.iterrows():
                patient_rows.append({"patient": patient, "model": model, "run": f"seed{s}",
                                      **row.to_dict()})

        predicted_pooled = pooled_p_final.argmax(axis=1)
        metrics_pooled, delta_u_pooled = region_level_metrics(predicted_pooled, data, f"{model}_pooled")
        seed_results.setdefault(model, {})["pooled"] = metrics_pooled
        delta_u_store[(model, "pooled")] = delta_u_pooled
        pu_pooled = patient_level_normalized_utility(delta_u_pooled, data)
        for patient, row in pu_pooled.iterrows():
            patient_rows.append({"patient": patient, "model": model, "run": "pooled", **row.to_dict()})

    patient_df = pd.DataFrame(patient_rows)
    patient_df.to_csv(REPO_ROOT / "PTDR_K6_PATIENT_RESULTS.csv", index=False)
    print(f"Wrote PTDR_K6_PATIENT_RESULTS.csv ({len(patient_df)} rows)")

    # ---- paired bootstrap: PRIMARY (pooled) + per-seed (consistency check) ----
    def get_series(model, run):
        sub = patient_df[(patient_df.model == model) & (patient_df.run == run)].set_index("patient")["normalized_net_utility"]
        return sub

    bootstrap_out = {}
    T_pooled, M_pooled = get_series("T", "pooled"), get_series("M", "pooled")
    common_idx = T_pooled.index.intersection(M_pooled.index)
    diff_pooled = (T_pooled.loc[common_idx] - M_pooled.loc[common_idx])
    bootstrap_out["primary_pooled"] = paired_bootstrap(diff_pooled)

    per_seed_direction = []
    for s in SEEDS:
        T_s, M_s = get_series("T", f"seed{s}"), get_series("M", f"seed{s}")
        idx = T_s.index.intersection(M_s.index)
        diff_s = T_s.loc[idx] - M_s.loc[idx]
        res = paired_bootstrap(diff_s)
        bootstrap_out[f"seed{s}"] = res
        per_seed_direction.append(res["favors"])

    n_seeds_favor_T = sum(1 for d in per_seed_direction if d == "T")
    n_seeds_favor_M = sum(1 for d in per_seed_direction if d == "M")
    bootstrap_out["seed_direction_summary"] = {
        "per_seed_favors": per_seed_direction, "n_seeds_favor_T": n_seeds_favor_T, "n_seeds_favor_M": n_seeds_favor_M,
        "typed_consistent_in_2_of_3_or_more": n_seeds_favor_T >= 2,
        "monolithic_consistent_in_2_of_3_or_more": n_seeds_favor_M >= 2,
    }
    (REPO_ROOT / "PTDR_K6_PAIRED_BOOTSTRAP.json").write_text(json.dumps(bootstrap_out, indent=2))
    print("Wrote PTDR_K6_PAIRED_BOOTSTRAP.json")
    print(json.dumps(bootstrap_out["primary_pooled"], indent=2))
    print(f"Per-seed direction: {per_seed_direction}")

    (REPO_ROOT / "PTDR_K6_SEED_RESULTS.json").write_text(json.dumps(seed_results, indent=2, default=str))
    print("Wrote PTDR_K6_SEED_RESULTS.json")

    # ---- PTDR_K6_CORRECTION_RESULTS.json: primary = pooled results for both models ----
    correction_out = {
        "T_pooled": seed_results["T"]["pooled"], "M_pooled": seed_results["M"]["pooled"],
        "note": "Primary correction-quality results use the pooled (3-seed-averaged probability, "
                "then argmax) prediction per architecture. Individual per-seed results are in "
                "PTDR_K6_SEED_RESULTS.json for the seed-wise consistency check (governing task section 19D).",
    }
    (REPO_ROOT / "PTDR_K6_CORRECTION_RESULTS.json").write_text(json.dumps(correction_out, indent=2, default=str))
    print("Wrote PTDR_K6_CORRECTION_RESULTS.json")

    # ---- RC/NETC mandatory strata markdown ----
    md = ["# PTDR_K6_RC_NETC_AUDIT", "", "Pooled (3-seed-averaged probability, argmax) results for both "
          "architectures. Per-seed breakdowns are in `PTDR_K6_SEED_RESULTS.json`.", ""]
    for target in ["rc", "netc"]:
        md.append(f"## {target.upper()} final-target corrections")
        md.append("")
        md.append("| Model | n | Beneficial | Harmful | Net | Ratio | Coverage | Patients (beneficial) |")
        md.append("|---|---|---|---|---|---|---|---|")
        for model in ["T", "M"]:
            r = seed_results[model]["pooled"]["by_predicted_final_target"].get(target)
            if r is None:
                md.append(f"| {model} | 0 | - | - | - | - | - | - |")
                continue
            md.append(f"| {model} | {r['n']} | {r['beneficial']} | {r['harmful']} | {r['net_corrected']} | "
                       f"{r['beneficial_to_harmful_ratio']} | {r.get('correction_coverage_of_target_actionable')} | "
                       f"{r['patient_support_beneficial']} |")
        md.append("")
    md.append("## NETC-source -> RC-target dedicated row (Stage 1.6's dominant harmful pathway)")
    md.append("")
    md.append("| Model | n | Beneficial | Harmful | Net | Ratio | Patients |")
    md.append("|---|---|---|---|---|---|---|")
    for model in ["T", "M"]:
        r = seed_results[model]["pooled"]["netc_source_to_rc_target_dedicated_row"]
        md.append(f"| {model} | {r['n']} | {r['beneficial']} | {r['harmful']} | {r['net_corrected']} | "
                   f"{r['beneficial_to_harmful_ratio']} | {r['patient_support']} |")
    md.append("")
    (REPO_ROOT / "PTDR_K6_RC_NETC_AUDIT.md").write_text("\n".join(md))
    print("Wrote PTDR_K6_RC_NETC_AUDIT.md")

    print("\n=== RC pooled summary ===")
    for model in ["T", "M"]:
        print(model, seed_results[model]["pooled"]["by_predicted_final_target"].get("rc"))
    print("\n=== NETC pooled summary ===")
    for model in ["T", "M"]:
        print(model, seed_results[model]["pooled"]["by_predicted_final_target"].get("netc"))
    print("\n=== NETC-source->RC pooled ===")
    for model in ["T", "M"]:
        print(model, seed_results[model]["pooled"]["netc_source_to_rc_target_dedicated_row"])


if __name__ == "__main__":
    main()
