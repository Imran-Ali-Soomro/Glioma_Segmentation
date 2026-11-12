"""PTDR Stage 1: OOF inference + region/candidate extraction for ONE fold.

Run once a fold's training has finished. Loads that fold's checkpoint,
performs tiled/sliding-window inference on that fold's VAL cases ONLY
(genuine out-of-fold predictions -- this checkpoint never trained on these
patients), entirely in the preprocessed 1mm-isotropic space (same space the
network itself operates in -- makes every subsequent distance/size
computation already in true millimeters, no spacing conversion needed).

Does NOT save dense per-voxel softmax volumes to disk (prohibitively large at
this scale, ~300GB+ for the full pool) -- everything needed downstream is
accumulated per-case into compact records during a single inference pass,
mirroring the 2D Phase-0.5/0.6 scripts' design, now in true 3D with a 3D
connected-component structure and physical-mm distances.

Usage:
    python oof_predict_and_extract.py --fold 0
"""
import argparse
import json
import pydoc
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1])); from mriseg_paths import P as _P, CODE as _CODE  # noqa: E402
REPO_ROOT = _P(r"")
NNUNET_DATA = REPO_ROOT / "nnUNet_data"
DATASET = "Dataset001_MUGliomaPost"
PLANS_NAME = "nnUNetResEncUNetMPlans"
TRAINER_NAME = "nnUNetTrainerPTDROOF_100epochs"
CLASS_NAMES = ["bg", "netc", "snfh", "et", "rc"]
NUM_CLASSES = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MIN_COMPONENT_SIZE_VOXELS = 8  # 3D analog of the 2D >=4-voxel floor (roughly volume-matched
                                # order of magnitude for a small 3D blob; documented, not tuned)

sys.path.insert(0, str(REPO_ROOT / "nnunet_ext"))
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetBlosc2  # noqa: E402


def build_network(plans: dict, dataset_json: dict, device):
    from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet
    config = plans["configurations"]["3d_fullres"]
    arch = config["architecture"]
    arch_kwargs = dict(arch["arch_kwargs"])
    for key in arch["_kw_requires_import"]:
        if arch_kwargs[key] is not None:
            arch_kwargs[key] = pydoc.locate(arch_kwargs[key])
    num_classes = len(dataset_json["labels"])
    net = ResidualEncoderUNet(
        input_channels=4, num_classes=num_classes, n_stages=arch_kwargs["n_stages"],
        features_per_stage=arch_kwargs["features_per_stage"], conv_op=arch_kwargs["conv_op"],
        kernel_sizes=arch_kwargs["kernel_sizes"], strides=arch_kwargs["strides"],
        n_blocks_per_stage=arch_kwargs["n_blocks_per_stage"],
        n_conv_per_stage_decoder=arch_kwargs["n_conv_per_stage_decoder"],
        conv_bias=arch_kwargs["conv_bias"], norm_op=arch_kwargs["norm_op"], norm_op_kwargs=arch_kwargs["norm_op_kwargs"],
        dropout_op=arch_kwargs["dropout_op"], dropout_op_kwargs=arch_kwargs["dropout_op_kwargs"],
        nonlin=arch_kwargs["nonlin"], nonlin_kwargs=arch_kwargs["nonlin_kwargs"], deep_supervision=False,
    ).to(device)
    return net, tuple(config["patch_size"])


def pad_to_multiple(arr: np.ndarray, patch_size, pad_value=0.0):
    c, x, y, z = arr.shape
    px = int(np.ceil(x / patch_size[0])) * patch_size[0]
    py = int(np.ceil(y / patch_size[1])) * patch_size[1]
    pz = int(np.ceil(z / patch_size[2])) * patch_size[2]
    padded = np.full((c, px, py, pz), pad_value, dtype=arr.dtype)
    padded[:, :x, :y, :z] = arr
    return padded, (x, y, z)


class DecoderFeatureCapture:
    """Predeclared, frozen S2 representation source (protocol sec 11): the final
    full-resolution decoder feature tensor immediately before the segmentation-logit
    projection -- i.e. the output of net.decoder.stages[-1], captured via a forward hook.
    No feature-shopping across other encoder/decoder levels."""

    def __init__(self, net):
        self.feat = None
        self.handle = net.decoder.stages[-1].register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        self.feat = output.detach()

    def remove(self):
        self.handle.remove()


@torch.no_grad()
def infer_volume_softmax(net, data: np.ndarray, patch_size, device, feat_capture=None):
    """Non-overlapping tiled inference (matches this project's established 3D analysis
    pattern). Returns full-volume softmax (5,X,Y,Z) and, if feat_capture is given, the
    full-volume pre-logit decoder feature tensor (C,X,Y,Z) -- both in the preprocessed
    1mm space."""
    padded, orig_shape = pad_to_multiple(data, patch_size, pad_value=0.0)
    _, px, py, pz = padded.shape
    probs_full = np.zeros((NUM_CLASSES, px, py, pz), dtype=np.float32)
    feat_full = None
    for xi in range(0, px, patch_size[0]):
        for yi in range(0, py, patch_size[1]):
            for zi in range(0, pz, patch_size[2]):
                patch = padded[:, xi:xi + patch_size[0], yi:yi + patch_size[1], zi:zi + patch_size[2]]
                x = torch.from_numpy(patch).unsqueeze(0).float().to(device)
                with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                    logits = net(x)
                probs = F.softmax(logits.float(), dim=1).squeeze(0).cpu().numpy()
                probs_full[:, xi:xi + patch_size[0], yi:yi + patch_size[1], zi:zi + patch_size[2]] = probs
                if feat_capture is not None:
                    feat = feat_capture.feat.float().squeeze(0).cpu().numpy()  # (C, pz0,py0,pz0)
                    if feat_full is None:
                        feat_full = np.zeros((feat.shape[0], px, py, pz), dtype=np.float32)
                    feat_full[:, xi:xi + patch_size[0], yi:yi + patch_size[1], zi:zi + patch_size[2]] = feat
    x, y, z = orig_shape
    probs_out = probs_full[:, :x, :y, :z]
    feat_out = feat_full[:, :x, :y, :z] if feat_full is not None else None
    return probs_out, feat_out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fold", type=int, required=True)
    args = p.parse_args()
    fold = args.fold

    preprocessed_dir = NNUNET_DATA / "nnUNet_preprocessed" / DATASET
    plans = json.loads((preprocessed_dir / f"{PLANS_NAME}.json").read_text())
    dataset_json = json.loads((preprocessed_dir / "dataset.json").read_text())
    derived = json.loads((preprocessed_dir / "PTDR_3D_OOF_SPLITS_DERIVED.json").read_text())
    val_cases = derived[fold]["val"]
    train_cases = derived[fold]["train"]
    print(f"Fold {fold}: {len(val_cases)} OOF val cases (never trained on by this checkpoint)")

    ckpt_path = (NNUNET_DATA / "nnUNet_results" / DATASET /
                f"{TRAINER_NAME}__{PLANS_NAME}__3d_fullres" / f"fold_{fold}" / "checkpoint_final.pth")
    assert ckpt_path.exists(), f"checkpoint not found: {ckpt_path}"

    data_folder = preprocessed_dir / "nnUNetPlans_3d_fullres"
    val_ds = nnUNetDatasetBlosc2(str(data_folder), identifiers=val_cases)

    net, patch_size = build_network(plans, dataset_json, DEVICE)
    ckpt = torch.load(str(ckpt_path), map_location=DEVICE, weights_only=False)
    net.load_state_dict(ckpt["network_weights"])
    net.eval()
    print(f"Loaded checkpoint: epoch={ckpt.get('current_epoch')}, patch_size={patch_size}")
    feat_capture = DecoderFeatureCapture(net)

    # ---- accumulators (this fold's contribution to the pooled OOF analysis) ----
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    region_records = []       # predicted-FG 3D connected components
    missed_components = []    # (seg==c) & (pred==bg) 3D connected components
    bg_score_accum = defaultdict(lambda: {
        "proposal_voxels": 0, "true_error_voxels": 0, "n_regions": 0, "sizes": [],
        "patients": set(), "recovered_missed_ids": set(),
    })

    import re
    def patient_of(cid): return re.match(r"(PatientID_\d+)_Timepoint_\d+", cid).group(1)

    SUMFG_T = [0.05, 0.10, 0.15, 0.20, 0.30]
    MAXFG_T = [0.05, 0.10, 0.15, 0.20, 0.30]
    MARGIN_T = [-0.50, -0.30, -0.20, -0.10, -0.05]
    SHELL_RADII_MM = [1, 2, 4, 8]  # == mm directly, isotropic 1mm preprocessed space

    t0 = time.time()
    for i, cid in enumerate(val_cases):
        pid = patient_of(cid)
        data, seg, _, properties = val_ds.load_case(cid)
        data = np.asarray(data)  # (4,X,Y,Z)
        seg = np.asarray(seg)[0].astype(np.int64)  # (X,Y,Z)

        probs, decoder_feat = infer_volume_softmax(net, data, patch_size, DEVICE, feat_capture)  # (5,X,Y,Z), (C,X,Y,Z)
        pred = probs.argmax(axis=0)

        for pcls in range(NUM_CLASSES):
            for gcls in range(NUM_CLASSES):
                confusion[pcls, gcls] += int(np.sum((pred == pcls) & (seg == gcls)))

        bg_mask = pred == 0
        fg_pred_mask = ~bg_mask
        dist_mm = ndimage.distance_transform_edt(bg_mask) if fg_pred_mask.any() else np.full(bg_mask.shape, 9999.0)
        error_mask_all = bg_mask & (seg != 0)

        max_fg = probs[1:5].max(axis=0)
        sum_fg = 1.0 - probs[0]
        margin = max_fg - probs[0]

        # ---- predicted-FG 3D regions ----
        # PERFORMANCE NOTE: all per-component work below is done via scipy.ndimage.find_objects()
        # bounding-box localization (comp_mask built only within the component's own small bbox)
        # and np.bincount/ndimage.sum vectorized aggregation, NOT by reconstructing a full-volume
        # boolean mask per component. An earlier version did `labeled == comp_id` inside the
        # per-component loop -- O(n_components x full_volume) -- which was observed to effectively
        # hang on cases where a lenient low-confidence threshold produced tens of thousands of tiny
        # fragmented proposal components (up to 135 threshold configs/case in the BG-recovery sweep
        # below). This rewrite computes the exact same component definitions, size filter, and
        # per-component statistics -- it changes only the computational cost, not the result.
        struct3d = ndimage.generate_binary_structure(3, 1)  # 6-connectivity, conservative/standard
        for c in range(1, NUM_CLASSES):
            mask = pred == c
            if not mask.any():
                continue
            labeled, n_comp = ndimage.label(mask, structure=struct3d)
            if n_comp == 0:
                continue
            sizes = np.bincount(labeled.ravel(), minlength=n_comp + 1)
            objects = ndimage.find_objects(labeled)
            for comp_id in range(1, n_comp + 1):
                size = int(sizes[comp_id])
                if size < MIN_COMPONENT_SIZE_VOXELS:
                    continue
                slc = objects[comp_id - 1]
                comp_mask_local = labeled[slc] == comp_id
                seg_local = seg[slc]
                probs_local = probs[(slice(None),) + slc]
                feat_local = decoder_feat[(slice(None),) + slc]
                class_counts = [int(np.sum(seg_local[comp_mask_local] == k)) for k in range(NUM_CLASSES)]
                mean_probs = [float(probs_local[k][comp_mask_local].mean()) for k in range(NUM_CLASSES)]
                # S2 predeclared feature: mean + std of the final pre-logit decoder feature
                # tensor pooled over this region's footprint (sec 11) -- NOT feature-shopped.
                feat_mean = [float(feat_local[k][comp_mask_local].mean()) for k in range(decoder_feat.shape[0])]
                feat_std = [float(feat_local[k][comp_mask_local].std()) for k in range(decoder_feat.shape[0])]
                region_records.append({
                    "patient": pid, "case": cid, "fold": fold, "pred_class": c,
                    "size_voxels_mm3": size, "class_voxel_counts": class_counts, "mean_probs": mean_probs,
                    "decoder_feat_mean": feat_mean, "decoder_feat_std": feat_std,
                })

        # ---- missed 3D components: (seg==c) & bg_mask ----
        missed_this_case = []  # list of dicts: id, slice (bbox), mask_local, size -- for the
                                # BG-recovery overlap check below, kept bbox-local for speed
        for c in range(1, NUM_CLASSES):
            fn_mask = (seg == c) & bg_mask
            if not fn_mask.any():
                continue
            labeled, n_comp = ndimage.label(fn_mask, structure=struct3d)
            if n_comp == 0:
                continue
            sizes = np.bincount(labeled.ravel(), minlength=n_comp + 1)
            objects = ndimage.find_objects(labeled)
            for comp_id in range(1, n_comp + 1):
                size = int(sizes[comp_id])
                if size < MIN_COMPONENT_SIZE_VOXELS:
                    continue
                slc = objects[comp_id - 1]
                local_mask = labeled[slc] == comp_id
                dist_local = dist_mm[slc]
                max_fg_local = max_fg[slc]
                margin_local = margin[slc]
                comp_id_global = f"{cid}_c{c}_comp{comp_id}"
                rec = {
                    "id": comp_id_global, "patient": pid, "case": cid, "fold": fold,
                    "gt_class": CLASS_NAMES[c], "size_voxels_mm3": size,
                    "min_distance_mm_to_pred_fg": float(dist_local[local_mask].min()),
                    "mean_fg_prob": float(max_fg_local[local_mask].mean()), "max_fg_prob": float(max_fg_local[local_mask].max()),
                    "mean_bg_margin": float(margin_local[local_mask].mean()),
                    "boundary_recoverable": bool(dist_local[local_mask].min() <= max(SHELL_RADII_MM)),
                    "comp_mask_nnz": size,
                }
                missed_components.append(rec)
                missed_this_case.append({"id": comp_id_global, "slice": slc, "mask_local": local_mask, "size": size})

        # ---- BG-recovery proposal sweep (whole-volume + shells), this case ----
        score_arrays = {"maxFG": max_fg, "sumFG": sum_fg, "margin": margin}
        threshold_sets = {"maxFG": MAXFG_T, "sumFG": SUMFG_T, "margin": MARGIN_T}
        configs = []
        for score_name, thresholds in threshold_sets.items():
            for T in thresholds:
                configs.append((score_name, T, None))
        for radius in SHELL_RADII_MM:
            for score_name, thresholds in threshold_sets.items():
                for T in thresholds:
                    configs.append((score_name, T, radius))

        for (score_name, T, radius) in configs:
            arr = score_arrays[score_name]
            base_mask = bg_mask if radius is None else (bg_mask & (dist_mm <= radius))
            if not base_mask.any():
                continue
            prop_mask = base_mask & (arr >= T)
            if not prop_mask.any():
                continue
            key = (score_name, T, radius)
            acc = bg_score_accum[key]
            labeled, n_comp = ndimage.label(prop_mask, structure=struct3d)
            if n_comp == 0:
                continue
            sizes = np.bincount(labeled.ravel(), minlength=n_comp + 1)
            valid_ids = np.nonzero(sizes[1:] >= MIN_COMPONENT_SIZE_VOXELS)[0] + 1  # exclude label 0 (background)
            if valid_ids.size == 0:
                continue
            # vectorized per-component sum of error_mask_all, restricted to the valid (size-passing)
            # component ids -- one optimized pass over the volume instead of N boolean-mask sums
            error_sums = ndimage.sum(error_mask_all, labeled, index=valid_ids)
            acc["n_regions"] += int(valid_ids.size)
            acc["sizes"].extend(int(s) for s in sizes[valid_ids])
            acc["proposal_voxels"] += int(sizes[valid_ids].sum())
            acc["true_error_voxels"] += int(np.sum(error_sums))
            acc["patients"].add(pid)
            # missed-component recovery: a missed component counts as recovered if ANY SINGLE
            # proposal component covers >=50% of its voxels (same semantics as before). Iterate
            # over missed components (few per case) and look up proposal labels only within each
            # missed component's own small bounding box, instead of looping over every proposal
            # component against every missed component at full-volume cost.
            if missed_this_case:
                valid_id_set = set(int(v) for v in valid_ids)
                for m in missed_this_case:
                    overlap_labels = labeled[m["slice"]][m["mask_local"]]
                    overlap_labels = overlap_labels[overlap_labels > 0]
                    if overlap_labels.size == 0:
                        continue
                    counts = np.bincount(overlap_labels)
                    for lbl in np.nonzero(counts)[0]:
                        if int(lbl) in valid_id_set and counts[lbl] / m["size"] >= 0.5:
                            acc["recovered_missed_ids"].add(m["id"])
                            break

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(val_cases)}] {time.time()-t0:.0f}s", flush=True)

    # ---- serialize this fold's contribution ----
    out_dir = REPO_ROOT / "analysis" / "ptdr_stage1" / "fold_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / f"fold{fold}_confusion.json").write_text(json.dumps({
        "fold": fold, "confusion_matrix": confusion.tolist(), "class_names": CLASS_NAMES,
        "n_val_cases": len(val_cases), "n_val_patients": len(set(patient_of(c) for c in val_cases)),
    }, indent=2))

    (out_dir / f"fold{fold}_regions.json").write_text(json.dumps(region_records))
    (out_dir / f"fold{fold}_missed_components.json").write_text(json.dumps(missed_components))

    bg_out = {}
    for (score_name, T, radius), acc in bg_score_accum.items():
        key_str = f"{score_name}_T{T}_shell{radius if radius is not None else 'whole'}"
        bg_out[key_str] = {
            "score": score_name, "threshold": T, "shell_radius_mm": radius,
            "n_regions": acc["n_regions"], "proposal_voxels": acc["proposal_voxels"],
            "true_error_voxels": acc["true_error_voxels"], "sizes": acc["sizes"],
            "n_patients_with_proposal": len(acc["patients"]),
            "n_missed_recovered": len(acc["recovered_missed_ids"]),
        }
    (out_dir / f"fold{fold}_bg_scores.json").write_text(json.dumps(bg_out))

    feat_capture.remove()
    print(f"\nFold {fold} DONE ({time.time()-t0:.0f}s total). "
          f"{len(region_records)} FG regions, {len(missed_components)} missed components.")
    print(f"Wrote fold{fold}_confusion.json / _regions.json / _missed_components.json / _bg_scores.json")


if __name__ == "__main__":
    main()
