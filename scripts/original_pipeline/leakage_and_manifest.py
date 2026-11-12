"""PTDR Stage 1, governing-task sections 5-7: hard-provenance leakage audit,
OOF completeness verification, and the frozen pre-analysis manifest.

Run AFTER oof_predict_and_extract.py --fold N has been run for all 5 folds
(produces analysis/ptdr_stage1/fold_outputs/fold{N}_*.json), and BEFORE
aggregate_k1.py / oracle_and_k2k3k4.py / k5_and_bg_recovery_and_geometry.py.

Design note on "prediction path" / "probability path" (governing task sec 5):
oof_predict_and_extract.py deliberately does NOT persist dense per-voxel softmax
or prediction volumes to disk (documented in that script's docstring: ~300GB+
for the full pool). Only pooled per-region/per-component summaries are kept.
This script's manifest entries therefore record prediction_path/probability_path
as null with an explicit "not_persisted_by_design" reason, rather than silently
omitting the field or fabricating a path. This is disclosed, not hidden.

Leakage check performed: for every realized region/missed-component record (i.e.
every record that actually appears in a fold's extracted output), assert the
record's patient is in that fold's VAL list and NOT in that fold's TRAIN list,
per PTDR_3D_OOF_SPLITS_DERIVED.json. This checks the ACTUAL realized extraction
output, not just the static split file (which was already separately audited
before any training began, in PTDR_3D_OOF_LEAKAGE_AUDIT.json from the pre-training
phase) -- i.e. it verifies oof_predict_and_extract.py actually used the right
checkpoint/val-case-list pairing for real, not just that the split file itself
was leak-free on paper.

Required output per governing task sec 5: leakage_detected = false.
If true: STOP with 3D_OOF_PROTOCOL_FAILURE, do not patch silently.
"""
import json
import re
from pathlib import Path

import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1])); from mriseg_paths import P as _P, CODE as _CODE  # noqa: E402
REPO_ROOT = _P(r"")
NNUNET_DATA = REPO_ROOT / "nnUNet_data"
DATASET = "Dataset001_MUGliomaPost"
PLANS_NAME = "nnUNetResEncUNetMPlans"
TRAINER_NAME = "nnUNetTrainerPTDROOF_100epochs"
RESULTS_BASE = NNUNET_DATA / "nnUNet_results" / DATASET / f"{TRAINER_NAME}__{PLANS_NAME}__3d_fullres"
FOLD_OUT = REPO_ROOT / "analysis" / "ptdr_stage1" / "fold_outputs"
N_FOLDS = 5


def patient_of(cid):
    return re.match(r"(PatientID_\d+)_Timepoint_\d+", cid).group(1)


def main():
    derived = json.loads((NNUNET_DATA / "nnUNet_preprocessed" / DATASET /
                           "PTDR_3D_OOF_SPLITS_DERIVED.json").read_text())

    # ---- 1. completeness: every fold produced its 4 output files, covering every intended val case ----
    completeness = {"folds": {}, "issues": []}
    all_case_to_folds = {}  # case -> list of folds whose regions/missed_components mention it
    expected_val_cases_by_fold = {f: set(derived[f]["val"]) for f in range(N_FOLDS)}
    union_expected_cases = set()
    for f in range(N_FOLDS):
        union_expected_cases |= expected_val_cases_by_fold[f]

    for f in range(N_FOLDS):
        fold_entry = {"expected_val_cases": len(expected_val_cases_by_fold[f])}
        conf_path = FOLD_OUT / f"fold{f}_confusion.json"
        reg_path = FOLD_OUT / f"fold{f}_regions.json"
        miss_path = FOLD_OUT / f"fold{f}_missed_components.json"
        bg_path = FOLD_OUT / f"fold{f}_bg_scores.json"
        missing_files = [p.name for p in (conf_path, reg_path, miss_path, bg_path) if not p.exists()]
        if missing_files:
            completeness["issues"].append(f"fold{f}: missing output files {missing_files}")
            fold_entry["status"] = "INCOMPLETE"
            completeness["folds"][f] = fold_entry
            continue

        conf = json.loads(conf_path.read_text())
        regions = json.loads(reg_path.read_text())
        missed = json.loads(miss_path.read_text())

        cases_in_regions = set(r["case"] for r in regions)
        cases_in_missed = set(m["case"] for m in missed)
        cases_seen = cases_in_regions | cases_in_missed
        # a val case with zero predicted-FG regions AND zero missed components is possible in
        # principle (perfect background-only prediction matching background-only GT) but should
        # be rare; we can only confirm a case was PROCESSED if it appears in confusion's n_val_cases
        # count, not necessarily in regions/missed (which can be legitimately empty for a case).
        fold_entry["n_val_cases_processed"] = conf["n_val_cases"]
        fold_entry["n_val_cases_expected"] = len(expected_val_cases_by_fold[f])
        fold_entry["cases_match_expected"] = (conf["n_val_cases"] == len(expected_val_cases_by_fold[f]))
        fold_entry["n_regions"] = len(regions)
        fold_entry["n_missed_components"] = len(missed)
        fold_entry["status"] = "COMPLETE" if fold_entry["cases_match_expected"] else "MISMATCH"
        if not fold_entry["cases_match_expected"]:
            completeness["issues"].append(
                f"fold{f}: processed {conf['n_val_cases']} cases, expected {len(expected_val_cases_by_fold[f])}")
        completeness["folds"][f] = fold_entry

        for c in cases_seen:
            all_case_to_folds.setdefault(c, []).append(f)

    # duplicate-case check: a case should never be the val case of more than one fold
    # (GroupKFold partitions by patient, folds are disjoint -- any overlap is a hard protocol error)
    duplicates = {c: folds for c, folds in all_case_to_folds.items() if len(set(folds)) > 1}
    if duplicates:
        completeness["issues"].append(f"DUPLICATE case(s) extracted by >1 fold: {duplicates}")

    # missing-case check: every case in the union of all folds' val lists should appear in some
    # fold's extracted output (via the n_val_cases_processed count check above); this is a coarse
    # second check against the raw union
    processed_union = set()
    for f in range(N_FOLDS):
        conf_path = FOLD_OUT / f"fold{f}_confusion.json"
        if conf_path.exists():
            processed_union |= expected_val_cases_by_fold[f]  # case was *attempted* if fold ran;
            # per-case success is already checked via n_val_cases_processed == n_val_cases_expected above
    missing_cases = union_expected_cases - processed_union
    if missing_cases:
        completeness["issues"].append(f"{len(missing_cases)} case(s) never processed by any fold: "
                                       f"{sorted(missing_cases)[:10]}{'...' if len(missing_cases) > 10 else ''}")

    completeness["total_union_expected_cases"] = len(union_expected_cases)
    completeness["duplicate_cases_detected"] = bool(duplicates)
    completeness["missing_cases_detected"] = bool(missing_cases)
    completeness["all_checks_passed"] = (
        not completeness["issues"]
        and all(completeness["folds"].get(f, {}).get("status") == "COMPLETE" for f in range(N_FOLDS))
    )

    (REPO_ROOT / "PTDR_3D_OOF_COMPLETENESS_SUMMARY.json").write_text(json.dumps(completeness, indent=2))
    print(f"Completeness: {'PASS' if completeness['all_checks_passed'] else 'ISSUES FOUND'}")
    for issue in completeness["issues"]:
        print(f"  ISSUE: {issue}")

    # ---- 2. hard-provenance leakage audit on REALIZED output ----
    leakage = {"leakage_detected": False, "violations": [], "folds_checked": []}
    for f in range(N_FOLDS):
        reg_path = FOLD_OUT / f"fold{f}_regions.json"
        miss_path = FOLD_OUT / f"fold{f}_missed_components.json"
        if not (reg_path.exists() and miss_path.exists()):
            continue
        train_patients = set(patient_of(c) for c in derived[f]["train"])
        val_patients = set(patient_of(c) for c in derived[f]["val"])
        assert not (train_patients & val_patients), f"fold{f}: train/val patient overlap in split file itself"

        regions = json.loads(reg_path.read_text())
        missed = json.loads(miss_path.read_text())
        n_checked = 0
        for rec in regions + missed:
            n_checked += 1
            pid = rec["patient"]
            if rec["fold"] != f:
                leakage["violations"].append(
                    {"fold_file": f, "record_fold_field": rec["fold"], "patient": pid, "case": rec["case"],
                     "reason": "record's internal fold field does not match the file it was extracted into"})
                continue
            if pid in train_patients:
                leakage["violations"].append(
                    {"fold": f, "patient": pid, "case": rec["case"],
                     "reason": f"patient {pid} was in fold{f}'s TRAIN set but appears in fold{f}'s extracted output"})
            elif pid not in val_patients:
                leakage["violations"].append(
                    {"fold": f, "patient": pid, "case": rec["case"],
                     "reason": f"patient {pid} is in neither fold{f}'s train nor val set (unexpected patient)"})
        leakage["folds_checked"].append({"fold": f, "n_records_checked": n_checked,
                                          "n_train_patients": len(train_patients), "n_val_patients": len(val_patients)})

    leakage["leakage_detected"] = len(leakage["violations"]) > 0
    (REPO_ROOT / "PTDR_3D_OOF_LEAKAGE_AUDIT.json").write_text(json.dumps(leakage, indent=2))
    print(f"\nLeakage audit: leakage_detected = {leakage['leakage_detected']}")
    if leakage["leakage_detected"]:
        print(f"  {len(leakage['violations'])} violation(s) found -- see PTDR_3D_OOF_LEAKAGE_AUDIT.json")
        print("  PROTOCOL FAILURE: STOP. Do not proceed to K1-K5. Do not patch silently.")
        return 1

    # ---- 3. frozen manifest (only built if completeness + leakage both pass) ----
    if not completeness["all_checks_passed"]:
        print("\nSkipping manifest: completeness checks did not pass. Fix completeness issues first.")
        return 1

    import hashlib
    import torch
    manifest = {"dataset": DATASET, "trainer": TRAINER_NAME, "plans": PLANS_NAME,
                "splits_file": "PTDR_3D_OOF_SPLITS_DERIVED.json", "folds": {}}
    for f in range(N_FOLDS):
        ckpt_path = RESULTS_BASE / f"fold_{f}" / "checkpoint_final.pth"
        val_summary_path = RESULTS_BASE / f"fold_{f}" / "validation" / "summary.json"
        ckpt_sha256 = None
        ckpt_epoch = None
        if ckpt_path.exists():
            h = hashlib.sha256()
            with open(ckpt_path, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
            ckpt_sha256 = h.hexdigest()
            ckpt_epoch = torch.load(str(ckpt_path), map_location="cpu", weights_only=False).get("current_epoch")
        val_dice = None
        if val_summary_path.exists():
            val_dice = json.loads(val_summary_path.read_text())["foreground_mean"]["Dice"]

        cases = []
        for cid in derived[f]["val"]:
            cases.append({
                "patient": patient_of(cid), "case": cid, "source_fold": f,
                "source_checkpoint": str(ckpt_path), "source_checkpoint_sha256": ckpt_sha256,
                "prediction_path": None, "probability_path": None,
                "persistence_note": "not_persisted_by_design -- oof_predict_and_extract.py pools "
                                     "region/component-level summaries in-memory per case and does not "
                                     "write dense per-voxel prediction/softmax volumes to disk "
                                     "(~300GB+ infeasible at this scale); pooled summaries are the "
                                     "frozen analysis unit for K1-K5, not raw volumes.",
            })
        manifest["folds"][f] = {
            "checkpoint_path": str(ckpt_path), "checkpoint_sha256": ckpt_sha256,
            "checkpoint_epoch": ckpt_epoch,
            "val_mean_dice": val_dice, "n_val_cases": len(derived[f]["val"]),
            "n_train_cases": len(derived[f]["train"]), "cases": cases,
        }
    manifest["total_oof_cases"] = sum(len(v["cases"]) for v in manifest["folds"].values())
    manifest["leakage_audit"] = "PASS (leakage_detected=false, see PTDR_3D_OOF_LEAKAGE_AUDIT.json)"
    manifest["completeness_audit"] = "PASS (see PTDR_3D_OOF_COMPLETENESS_SUMMARY.json)"
    manifest["frozen_note"] = ("K1-K5 must operate on this exact frozen dataset. No regeneration "
                                "after seeing results unless a genuine implementation error is found "
                                "and fully documented.")

    (REPO_ROOT / "PTDR_3D_OOF_MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nWrote PTDR_3D_OOF_MANIFEST.json ({manifest['total_oof_cases']} OOF cases across {N_FOLDS} folds)")
    print("All provenance/completeness/leakage checks PASSED. Safe to proceed to K1.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
