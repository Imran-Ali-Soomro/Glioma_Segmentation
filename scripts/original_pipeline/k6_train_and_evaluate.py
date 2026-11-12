"""PTDR K6, sections 4-9: train Model T (typed, factorized) and Model M
(monolithic) -- 3 seeds x 2 architectures x 5 frozen patient-grouped folds =
30 small MLP fits total, each training in seconds on ~15-16K rows x 75 S2
features. Patient-grouped cross-fitting uses EXACTLY the frozen `fold` column
(0-4) throughout -- no new GroupKFold, no reshuffling. Produces full
out-of-fold predictions for every region under every (architecture, seed).

Outputs: PTDR_K6_OOF_PREDICTIONS.npz (raw per-region OOF arrays, consumed by
the downstream k6_analyze_*.py scripts) and a training-log summary.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1])); from mriseg_paths import P as _P, CODE as _CODE  # noqa: E402
REPO_ROOT = _P(r"")
sys.path.insert(0, str(REPO_ROOT / "analysis" / "ptdr_stage1"))
from k6_data import load_k6_data, N_FOLDS, CLASS_NAMES  # noqa: E402
from k6_models import ModelT, ModelM, count_trainable_params  # noqa: E402

SEEDS = [0, 1, 2]
MAX_EPOCHS = 2000
PATIENCE = 50
LR = 1e-3
VAL_FRACTION_PATIENTS = 0.15
DEVICE = "cpu"


def patient_grouped_internal_split(train_patients, seed, fold):
    rng = np.random.default_rng(1000 * seed + fold)
    shuffled = rng.permutation(train_patients)
    n_val = max(1, int(round(VAL_FRACTION_PATIENTS * len(train_patients))))
    val_p = set(shuffled[:n_val].tolist())
    fit_p = set(shuffled[n_val:].tolist())
    return fit_p, val_p


def train_one_fold_model_T(Xtr_fit, ytr_fit, src_fit, Xval, yval, src_val, d_in, seed):
    torch.manual_seed(seed)
    model = ModelT(d_in).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    best_val, best_state, bad_epochs = np.inf, None, 0
    for epoch in range(MAX_EPOCHS):
        model.train()
        opt.zero_grad()
        loss = model.nll_loss(Xtr_fit, src_fit, ytr_fit)
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            vloss = model.nll_loss(Xval, src_val, yval).item()
        if vloss < best_val - 1e-5:
            best_val, best_state, bad_epochs = vloss, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, best_val, epoch + 1


def train_one_fold_model_M(Xtr_fit, ytr_fit, Xval, yval, d_in, seed):
    torch.manual_seed(seed)
    model = ModelM(d_in).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    best_val, best_state, bad_epochs = np.inf, None, 0
    for epoch in range(MAX_EPOCHS):
        model.train()
        opt.zero_grad()
        loss = model.nll_loss(Xtr_fit, ytr_fit)
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            vloss = model.nll_loss(Xval, yval).item()
        if vloss < best_val - 1e-5:
            best_val, best_state, bad_epochs = vloss, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, best_val, epoch + 1


def main():
    t0 = time.time()
    data = load_k6_data()
    assert data["n_label_violations"] == 0, "K6_PROTOCOL_FAILURE: y_final/label_act mechanical check failed"
    n = data["n"]
    d_in = data["S2"].shape[1]
    fold_arr = data["fold"]
    patients = data["patients"]
    pred_class = data["pred_class"]
    y_final = data["y_final"]

    results = {"T": {}, "M": {}}  # seed -> oof arrays
    training_log = []

    for seed in SEEDS:
        np.random.seed(seed)
        oof_pfinal_T = np.zeros((n, 5))
        oof_pact_T = np.zeros(n)
        oof_ptrans_T = np.zeros((n, 5))
        oof_pfinal_M = np.zeros((n, 5))

        for fold in range(N_FOLDS):
            train_mask = fold_arr != fold
            test_mask = fold_arr == fold
            train_patients = np.array(sorted(set(patients[train_mask])))
            fit_p, val_p = patient_grouped_internal_split(train_patients, seed, fold)
            fit_mask = train_mask & np.isin(patients, list(fit_p))
            val_mask = train_mask & np.isin(patients, list(val_p))

            scaler = StandardScaler().fit(data["S2"][train_mask])
            X_fit = scaler.transform(data["S2"][fit_mask])
            X_val = scaler.transform(data["S2"][val_mask])
            X_test = scaler.transform(data["S2"][test_mask])

            Xtr_fit_t = torch.tensor(X_fit, dtype=torch.float32, device=DEVICE)
            Xval_t = torch.tensor(X_val, dtype=torch.float32, device=DEVICE)
            Xtest_t = torch.tensor(X_test, dtype=torch.float32, device=DEVICE)
            ytr_fit_t = torch.tensor(y_final[fit_mask], dtype=torch.long, device=DEVICE)
            yval_t = torch.tensor(y_final[val_mask], dtype=torch.long, device=DEVICE)
            src_fit_t = torch.tensor(pred_class[fit_mask], dtype=torch.long, device=DEVICE)
            src_val_t = torch.tensor(pred_class[val_mask], dtype=torch.long, device=DEVICE)
            src_test_t = torch.tensor(pred_class[test_mask], dtype=torch.long, device=DEVICE)

            modelT, best_val_T, n_epochs_T = train_one_fold_model_T(
                Xtr_fit_t, ytr_fit_t, src_fit_t, Xval_t, yval_t, src_val_t, d_in, seed * 10 + fold)
            p_final_T, p_act_T, p_trans_T = modelT.predict_p_final(Xtest_t, src_test_t)
            oof_pfinal_T[test_mask] = p_final_T.cpu().numpy()
            oof_pact_T[test_mask] = p_act_T.cpu().numpy()
            oof_ptrans_T[test_mask] = p_trans_T.cpu().numpy()

            modelM, best_val_M, n_epochs_M = train_one_fold_model_M(
                Xtr_fit_t, ytr_fit_t, Xval_t, yval_t, d_in, seed * 10 + fold)
            p_final_M = modelM.predict_p_final(Xtest_t)
            oof_pfinal_M[test_mask] = p_final_M.cpu().numpy()

            log_entry = {"seed": seed, "fold": fold, "n_train_fit": int(fit_mask.sum()),
                         "n_train_val": int(val_mask.sum()), "n_test": int(test_mask.sum()),
                         "T_best_val_nll": best_val_T, "T_epochs": n_epochs_T,
                         "M_best_val_nll": best_val_M, "M_epochs": n_epochs_M}
            training_log.append(log_entry)
            print(f"seed={seed} fold={fold}: T val_nll={best_val_T:.4f} ({n_epochs_T} epochs) | "
                  f"M val_nll={best_val_M:.4f} ({n_epochs_M} epochs)")

        results["T"][seed] = {"p_final": oof_pfinal_T, "p_act": oof_pact_T, "p_trans": oof_ptrans_T}
        results["M"][seed] = {"p_final": oof_pfinal_M}

    # ---- save raw OOF arrays ----
    save_dict = {}
    for seed in SEEDS:
        save_dict[f"T_p_final_seed{seed}"] = results["T"][seed]["p_final"]
        save_dict[f"T_p_act_seed{seed}"] = results["T"][seed]["p_act"]
        save_dict[f"T_p_trans_seed{seed}"] = results["T"][seed]["p_trans"]
        save_dict[f"M_p_final_seed{seed}"] = results["M"][seed]["p_final"]
    np.savez_compressed(REPO_ROOT / "PTDR_K6_OOF_PREDICTIONS.npz", **save_dict)
    (REPO_ROOT / "PTDR_K6_TRAINING_LOG.json").write_text(json.dumps(training_log, indent=2))
    print(f"\nWrote PTDR_K6_OOF_PREDICTIONS.npz and PTDR_K6_TRAINING_LOG.json")
    print(f"Total training time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
