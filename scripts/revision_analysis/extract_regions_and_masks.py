"""Revision-1 extraction: for ONE base network (fold) and a chosen case set, run the frozen
nnU-Net checkpoint and save (a) region records identical in definition to the original
OOF extraction (3D connected components, 6-connectivity, >= 8 voxels, per predicted class),
now ALSO carrying (cc_class, cc_label) so a region can be mapped back to its exact connected
component in the saved prediction mask, and (b) the full predicted label volume + ground truth
in nnU-Net's preprocessed 1 mm isotropic space (for post-correction Dice / HD95).

No training. No change to any checkpoint. Reuses the original network/feature code.

Usage:
    python extract_regions_and_masks.py --set dev     --fold 0   # the fold's own OOF val cases
    python extract_regions_and_masks.py --set heldout --fold 0   # the 50 reserved patients' cases,
                                                                  # scored by base network 0
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1])); from mriseg_paths import P as _P, CODE as _CODE  # noqa: E402
REPO = _P(r"")
sys.path.insert(0, str(_CODE / "original_pipeline"))
from oof_predict_and_extract import (  # noqa: E402
    build_network, DecoderFeatureCapture, infer_volume_softmax, NNUNET_DATA, DATASET, PLANS_NAME,
    TRAINER_NAME, NUM_CLASSES, DEVICE, MIN_COMPONENT_SIZE_VOXELS,
)
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetBlosc2  # noqa: E402

OUT_ROOT = _P("Paper1_Revision/revision_analysis") / "outputs"


@torch.no_grad()
def infer_volume_sliding(net, data, patch_size, device, feat_capture, step=0.5):
    """nnU-Net-style Gaussian-weighted sliding window (50 % overlap, no mirroring); logits and the
    pre-logit decoder features are both accumulated with the same Gaussian weights."""
    import torch.nn.functional as F
    from acvl_utils.cropping_and_padding.padding import pad_nd_image
    from nnunetv2.inference.sliding_window_prediction import compute_gaussian, compute_steps_for_sliding_window
    padded, revert = pad_nd_image(torch.from_numpy(np.ascontiguousarray(data)).float(), patch_size, "constant", {"value": 0}, True, None)
    shape = padded.shape[1:]
    steps = compute_steps_for_sliding_window(shape, patch_size, step)
    gauss = compute_gaussian(tuple(patch_size), sigma_scale=1.0 / 8, value_scaling_factor=10, dtype=torch.float32, device=device)
    logit_acc = torch.zeros((NUM_CLASSES, *shape), dtype=torch.float32, device=device)
    feat_acc = None
    wsum = torch.zeros(shape, dtype=torch.float32, device=device)
    x_full = padded.float()
    for sx in steps[0]:
        for sy in steps[1]:
            for sz in steps[2]:
                sl = (slice(sx, sx + patch_size[0]), slice(sy, sy + patch_size[1]), slice(sz, sz + patch_size[2]))
                x = x_full[(slice(None),) + sl].unsqueeze(0).to(device)
                with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                    logits = net(x)
                feat = feat_capture.feat.float().squeeze(0)
                if feat_acc is None:
                    feat_acc = torch.zeros((feat.shape[0], *shape), dtype=torch.float32, device=device)
                logit_acc[(slice(None),) + sl] += logits.float().squeeze(0) * gauss
                feat_acc[(slice(None),) + sl] += feat * gauss
                wsum[sl] += gauss
    logit_acc /= wsum; feat_acc /= wsum
    probs = F.softmax(logit_acc, dim=0)
    sl_rev = tuple(revert[1:])
    return probs[(slice(None),) + sl_rev].cpu().numpy(), feat_acc[(slice(None),) + sl_rev].cpu().numpy()


def patient_of(cid):
    return re.match(r"(PatientID_\d+)_Timepoint_\d+", cid).group(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["dev", "heldout"], required=True)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--infer", choices=["tiled", "sliding"], default="tiled")
    args = ap.parse_args()
    out_root = (_P("Paper1_Revision/revision_analysis") / ("outputs_slide" if args.infer == "sliding" else "outputs"))
    fold = args.fold

    pre = NNUNET_DATA / "nnUNet_preprocessed" / DATASET
    plans = json.loads((pre / f"{PLANS_NAME}.json").read_text())
    dataset_json = json.loads((pre / "dataset.json").read_text())

    if args.set == "dev":
        derived = json.loads((pre / "PTDR_3D_OOF_SPLITS_DERIVED.json").read_text())
        cases = derived[fold]["val"]
    else:
        assign = json.loads((REPO / "split_v2" / "PATIENT_ASSIGNMENT.json").read_text())
        ho = set(assign["held_out_patients"])
        all_ids = sorted(p.name[:-4] for p in (pre / "nnUNetPlans_3d_fullres").glob("*.pkl"))
        cases = [c for c in all_ids if patient_of(c) in ho]
        # hard guard: no held-out case may belong to any base-network training set
        derived = json.loads((pre / "PTDR_3D_OOF_SPLITS_DERIVED.json").read_text())
        dev_all = set(derived[fold]["train"]) | set(derived[fold]["val"])
        assert not (set(cases) & dev_all), "held-out/dev overlap"
    print(f"set={args.set} fold={fold}: {len(cases)} cases, {len(set(patient_of(c) for c in cases))} patients", flush=True)

    ckpt_path = (NNUNET_DATA / "nnUNet_results" / DATASET / f"{TRAINER_NAME}__{PLANS_NAME}__3d_fullres"
                 / f"fold_{fold}" / "checkpoint_final.pth")
    net, patch_size = build_network(plans, dataset_json, DEVICE)
    ckpt = torch.load(str(ckpt_path), map_location=DEVICE, weights_only=False)
    net.load_state_dict(ckpt["network_weights"])
    net.eval()
    cap = DecoderFeatureCapture(net)
    ds = nnUNetDatasetBlosc2(str(pre / "nnUNetPlans_3d_fullres"), identifiers=cases)

    out_dir = out_root / args.set / f"fold{fold}"
    (out_dir / "masks").mkdir(parents=True, exist_ok=True)
    struct3d = ndimage.generate_binary_structure(3, 1)
    records = []
    t0 = time.time()
    for i, cid in enumerate(cases):
        data, seg, _, _ = ds.load_case(cid)
        data = np.asarray(data)
        seg = np.asarray(seg)[0].astype(np.int64)
        probs, feat = (infer_volume_sliding if args.infer == "sliding" else infer_volume_softmax)(net, data, patch_size, DEVICE, cap)
        pred = probs.argmax(axis=0)
        pid = patient_of(cid)
        for c in range(1, NUM_CLASSES):
            mask = pred == c
            if not mask.any():
                continue
            labeled, n_comp = ndimage.label(mask, structure=struct3d)
            sizes = np.bincount(labeled.ravel(), minlength=n_comp + 1)
            objects = ndimage.find_objects(labeled)
            for comp_id in range(1, n_comp + 1):
                size = int(sizes[comp_id])
                if size < MIN_COMPONENT_SIZE_VOXELS:
                    continue
                slc = objects[comp_id - 1]
                m = labeled[slc] == comp_id
                seg_l, probs_l, feat_l = seg[slc], probs[(slice(None),) + slc], feat[(slice(None),) + slc]
                records.append({
                    "patient": pid, "case": cid, "base_network": fold, "pred_class": c,
                    "cc_class": c, "cc_label": int(comp_id),
                    "size_voxels_mm3": size,
                    "class_voxel_counts": [int(np.sum(seg_l[m] == k)) for k in range(NUM_CLASSES)],
                    "mean_probs": [float(probs_l[k][m].mean()) for k in range(NUM_CLASSES)],
                    "decoder_feat_mean": [float(feat_l[k][m].mean()) for k in range(feat.shape[0])],
                    "decoder_feat_std": [float(feat_l[k][m].std()) for k in range(feat.shape[0])],
                })
        np.savez_compressed(out_dir / "masks" / f"{cid}.npz", pred=pred.astype(np.uint8), seg=seg.astype(np.int8))
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(cases)}] {time.time()-t0:.0f}s", flush=True)
    cap.remove()
    (out_dir / "regions.json").write_text(json.dumps(records))
    print(f"DONE set={args.set} fold={fold}: {len(records)} regions, {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
