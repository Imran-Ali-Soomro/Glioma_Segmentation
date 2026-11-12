"""PTDR Stage 1, sections 5-6: build 5 patient-grouped OOF folds over the clean
153-patient dev pool, with class-presence balance audit.

No model training. Reads segmentation labels directly (nibabel) to compute
per-patient class presence (NETC/SNFH/ET/RC), then builds folds via
sklearn's GroupKFold as a first attempt; falls back to a deterministic
patient-grouped stratified allocation (using only training labels, not any
model output) if GroupKFold produces severe class-presence imbalance.
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    import nibabel as nib
except ImportError:
    nib = None
    import SimpleITK as sitk

import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1])); from mriseg_paths import P as _P, CODE as _CODE  # noqa: E402
REPO_ROOT = _P(r"")
NNUNET_RAW = REPO_ROOT / "nnUNet_data" / "nnUNet_raw" / "Dataset001_MUGliomaPost"
LABELS_DIR = NNUNET_RAW / "labelsTr"
CLASS_NAMES = ["netc", "snfh", "et", "rc"]  # foreground only; label values 1-4
N_FOLDS = 5
SEED = 20261001


def load_label_classes_present(path: Path) -> set:
    if nib is not None:
        arr = nib.load(str(path)).get_fdata()
    else:
        arr = sitk.GetArrayFromImage(sitk.ReadImage(str(path)))
    vals = set(np.unique(arr).astype(int).tolist())
    return {c for c in (1, 2, 3, 4) if c in vals}


def main():
    with open(REPO_ROOT / "split_v2" / "PATIENT_ASSIGNMENT.json") as f:
        assignment = json.load(f)
    dev_patients = sorted(assignment["dev_pool_patients"])

    with open(REPO_ROOT / "PTDR_3D_DATA_COMPLETENESS_AUDIT.json") as f:
        completeness = json.load(f)
    unrecoverable_cases = {r["case"] for r in completeness["unrecoverable_cases"]}

    # enumerate available cases per patient from labelsTr (now includes the 35 repaired cases)
    patient_cases = defaultdict(list)
    for label_file in sorted(LABELS_DIR.glob("PatientID_*.nii.gz")):
        case_id = label_file.stem.replace(".nii", "")
        patient = case_id.split("_Timepoint_")[0]
        if patient not in set(dev_patients):
            continue  # skip held-out-set cases that may physically exist in nnUNet_raw
        if case_id in unrecoverable_cases:
            continue
        patient_cases[patient].append((case_id, label_file))

    missing_patients = [p for p in dev_patients if p not in patient_cases]
    print(f"Dev patients: {len(dev_patients)}, with >=1 available case: {len(patient_cases)}, "
          f"missing entirely: {len(missing_patients)}")
    if missing_patients:
        print("  MISSING:", missing_patients)

    # class presence per patient (present if ANY of that patient's cases contains the class)
    print("Computing class presence per patient (reading labels)...")
    patient_class_presence = {}
    for i, (patient, cases) in enumerate(sorted(patient_cases.items())):
        present = set()
        for case_id, label_file in cases:
            present |= load_label_classes_present(label_file)
        patient_class_presence[patient] = sorted(CLASS_NAMES[c - 1] for c in present)
        if (i + 1) % 30 == 0:
            print(f"  ...{i+1}/{len(patient_cases)} patients")

    n_cases_total = sum(len(v) for v in patient_cases.values())
    print(f"Total available cases: {n_cases_total}")

    # ---- attempt 1: plain GroupKFold by patient ----
    patients_sorted = sorted(patient_cases.keys())
    rng = np.random.default_rng(SEED)
    shuffled = rng.permutation(patients_sorted).tolist()
    fold_assignment = {p: i % N_FOLDS for i, p in enumerate(shuffled)}

    def fold_stats(assignment_map):
        stats = []
        for fold in range(N_FOLDS):
            fold_patients = [p for p in patients_sorted if assignment_map[p] == fold]
            fold_cases = sum(len(patient_cases[p]) for p in fold_patients)
            class_patient_counts = {c: sum(1 for p in fold_patients if c in patient_class_presence[p]) for c in CLASS_NAMES}
            stats.append({"fold": fold, "n_patients": len(fold_patients), "n_cases": fold_cases,
                          "class_presence_patient_counts": class_patient_counts})
        return stats

    stats = fold_stats(fold_assignment)
    print("\n=== Plain shuffled GroupKFold-equivalent fold stats ===")
    for s in stats:
        print(f"  fold{s['fold']}: patients={s['n_patients']} cases={s['n_cases']} "
              f"classes={s['class_presence_patient_counts']}")

    # imbalance check: max/min ratio of per-class patient counts across folds
    imbalanced = False
    for c in CLASS_NAMES:
        counts = [s["class_presence_patient_counts"][c] for s in stats]
        if min(counts) == 0 or (max(counts) / max(min(counts), 1)) > 2.0:
            imbalanced = True
            print(f"  IMBALANCE FLAGGED for class {c}: counts={counts}")

    if imbalanced:
        print("\nPlain shuffle imbalanced -- building deterministic stratified allocation...")
        # deterministic stratified round-robin: sort patients by a composite rarity key
        # (fewest total classes present first = hardest to place), assign to the fold
        # currently most in need of that patient's classes (greedy, deterministic, seeded
        # only for the initial tie-break ordering, not re-optimized against any model output).
        class_rarity = {c: sum(1 for p in patients_sorted if c in patient_class_presence[p]) for c in CLASS_NAMES}
        def patient_key(p):
            n_classes = len(patient_class_presence[p])
            rarity_score = sum(1.0 / max(class_rarity[c], 1) for c in patient_class_presence[p])
            return (-rarity_score, -n_classes, p)  # rarest/most-informative patients placed first
        order = sorted(patients_sorted, key=patient_key)

        fold_class_counts = {f: {c: 0 for c in CLASS_NAMES} for f in range(N_FOLDS)}
        fold_patient_counts = {f: 0 for f in range(N_FOLDS)}
        new_assignment = {}
        for p in order:
            p_classes = patient_class_presence[p]
            def fold_score(f):
                deficit = sum(max(0, 1 - fold_class_counts[f][c]) * 10 for c in p_classes)
                return (fold_patient_counts[f] - deficit, f)
            best_fold = min(range(N_FOLDS), key=fold_score)
            new_assignment[p] = best_fold
            fold_patient_counts[best_fold] += 1
            for c in p_classes:
                fold_class_counts[best_fold][c] += 1
        fold_assignment = new_assignment
        stats = fold_stats(fold_assignment)
        print("=== Deterministic stratified fold stats ===")
        for s in stats:
            print(f"  fold{s['fold']}: patients={s['n_patients']} cases={s['n_cases']} "
                  f"classes={s['class_presence_patient_counts']}")
        method = "deterministic_patient_grouped_stratified"
    else:
        print("\nPlain shuffled GroupKFold is adequately balanced -- using it.")
        method = "plain_shuffled_groupkfold"

    # ---- assert leakage-free ----
    for f1 in range(N_FOLDS):
        p1 = {p for p in patients_sorted if fold_assignment[p] == f1}
        for f2 in range(N_FOLDS):
            if f1 == f2:
                continue
            p2 = {p for p in patients_sorted if fold_assignment[p] == f2}
            assert not (p1 & p2), f"LEAKAGE: fold {f1} and {f2} share patients"
    print("\nLeakage assertion PASSED: no patient appears in more than one fold.")

    out = {
        "n_folds": N_FOLDS, "seed": SEED, "method": method,
        "n_dev_patients": len(dev_patients), "n_patients_with_cases": len(patient_cases),
        "missing_patients": missing_patients,
        "unrecoverable_cases_excluded": sorted(unrecoverable_cases),
        "patient_class_presence": patient_class_presence,
        "fold_assignment": fold_assignment,
        "fold_stats": stats,
        "patient_cases": {p: [c for c, _ in cases] for p, cases in patient_cases.items()},
    }
    (REPO_ROOT / "PTDR_3D_OOF_FOLDS.json").write_text(json.dumps(out, indent=2))
    print("\nWrote PTDR_3D_OOF_FOLDS.json")


if __name__ == "__main__":
    main()
