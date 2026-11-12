"""R1.10 / R2.7: segmentation-level effect of correction.  Rebuilds each corrected label map by applying
a variant's per-region decisions to the saved base-network predictions (exact connected components),
then scores Dice, HD95 (mm; preprocessed 1 mm isotropic grid) and predicted-vs-reference volume per class
against ground truth.  Ground-truth sentinel voxels (-1, outside the non-zero image mask) are treated as
background.  Usage:  python a6_dice_hd95.py --set dev|heldout"""
import argparse
import json
import sys
from multiprocessing import Pool

import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1])); from mriseg_paths import P as _P, CODE as _CODE  # noqa: E402
sys.path.insert(0, str(_CODE / "revision_analysis"))
from rev_common import *  # noqa
from scipy import ndimage

STRUCT = ndimage.generate_binary_structure(3, 1)
FG = [1, 2, 3, 4]


def hd95(a, b):
    if not a.any() or not b.any():
        return np.nan
    idx = np.argwhere(a | b)
    lo = np.maximum(idx.min(0) - 1, 0); hi = idx.max(0) + 2
    sl = tuple(slice(int(l), int(h)) for l, h in zip(lo, hi))
    a, b = a[sl], b[sl]
    sa = a & ~ndimage.binary_erosion(a, STRUCT); sb = b & ~ndimage.binary_erosion(b, STRUCT)
    da = ndimage.distance_transform_edt(~sa); db = ndimage.distance_transform_edt(~sb)
    return float(np.percentile(np.concatenate([db[sa], da[sb]]), 95))


def dice(a, b):
    sa, sb = a.sum(), b.sum()
    if sa == 0 and sb == 0:
        return np.nan
    return float(2 * (a & b).sum() / (sa + sb))


def work(args):
    case, mask_path, fresh_regions, variants = args      # variants: {name: [final_class per region, in order]}
    z = np.load(mask_path); pred = z["pred"].astype(np.int64); seg = z["seg"].astype(np.int64)
    seg = np.where(seg < 0, 0, seg)
    labs, objs = {}, {}
    for c in FG:
        m = pred == c
        if m.any():
            labs[c], _ = ndimage.label(m, structure=STRUCT)
            objs[c] = ndimage.find_objects(labs[c])
    rows, base_cache = [], {}
    for vname, finals in variants.items():
        new = pred.copy()
        for r, fin in zip(fresh_regions, finals):
            if fin != r["pred_class"]:
                sl = objs[r["cc_class"]][r["cc_label"] - 1]
                loc = labs[r["cc_class"]][sl] == r["cc_label"]
                new[sl][loc] = fin
        for c in FG:
            a, b = new == c, seg == c
            key = (c,)
            if vname != "baseline" and np.array_equal(a, base_cache.get(("m", c))):
                h = base_cache[("h", c)]
            else:
                h = hd95(a, b)
            if vname == "baseline":
                base_cache[("m", c)] = a; base_cache[("h", c)] = h
            rows.append({"case": case, "variant": vname, "cls": CLASS_NAMES[c], "dice": dice(a, b), "hd95": h,
                         "vol_pred": int(a.sum()), "vol_gt": int(b.sum())})
    return rows


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--set", default="dev"); ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()
    D = np.load(OUT / ("A3_VARIANT_DECISIONS.npz"))
    names = {"baseline": "src", "P0_typed": "P0_unw__T", "P0_mono": "P0_unw__M", "P2_typed": "P2_unw__T",
             "P2_mono": "P2_unw__M", "P2_mono_matchedK": "P2_unw__M_matchedK", "oracle_relabel": "oracle_target"}
    if args.set == "heldout":
        names = {k: v for k, v in names.items()}
    d = load_dev() if args.set == "dev" else None
    jobs = []
    if args.set == "dev":
        for f in range(5):
            fresh = json.loads((OUT / "dev" / f"fold{f}" / "regions.json").read_text())
            by_case = {}
            for r in fresh:
                by_case.setdefault(r["case"], []).append(r)
            for case, regs in by_case.items():
                idx = np.flatnonzero(d["cases"] == case)
                # exact alignment check between fresh regions and the saved region table
                assert len(idx) == len(regs), f"{case}: {len(idx)} saved vs {len(regs)} fresh regions"
                for i, r in zip(idx, regs):
                    assert d["pred_class"][i] == r["pred_class"] and int(d["size_voxels"][i]) == r["size_voxels_mm3"], case
                jobs.append((case, OUT / "dev" / f"fold{f}" / "masks" / f"{case}.npz", regs,
                             {v: [int(D[k][i]) for i in idx] for v, k in names.items()}))
    else:
        raise SystemExit("heldout mode is driven by h3_heldout_eval.py")
    have = {j[0] for j in jobs}                    # cases with no candidate region: unchanged by any correction, but they
    for f in range(5):                              # belong to the baseline, so they are scored (no edits) as well
        for p in sorted((OUT / "dev" / f"fold{f}" / "masks").glob("*.npz")):
            if p.stem not in have:
                jobs.append((p.stem, p, [], {v: [] for v in names}))
    print("cases:", len(jobs), flush=True)
    with Pool(args.workers) as p:
        res = p.map(work, jobs, chunksize=4)
    rows = [r for rr in res for r in rr]
    df = pd.DataFrame(rows)
    df["patient"] = df["case"].str.extract(r"(PatientID_\d+)")[0]
    df.to_csv(OUT / f"A6_DICE_HD95_{args.set}.csv", index=False)
    print("rows", len(df), "saved")


if __name__ == "__main__":
    main()
