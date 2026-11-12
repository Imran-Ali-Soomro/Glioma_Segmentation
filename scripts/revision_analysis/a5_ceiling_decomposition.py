"""R2.9: quantify what produces the gap between the 69.9 % of error that is not missed tissue and the
42.88 % foreground-relabelling ceiling.  Exact identities (all voxel counts, ground-truth classes 0-4 only):

  E_valid                         total prediction-vs-GT errors            (8,662,107)
  missed                          predicted BG, GT foreground              (2,607,924)  -> unreachable
  non-missed error N = E - missed                                           (6,054,183)
     = E_in   error inside candidate regions (>= 8 voxels)
     + F      error inside predicted-FG components < 8 voxels
  E_in = gain + H                  gain = sum_r [max_a U - U(c_r)]  (the 42.88 % ceiling numerator)
                                   H    = sum_r [sum_a U - max_a U] (heterogeneity: voxels that whole-region
                                          relabelling cannot fix even with the oracle target)
  F   = gain_sub8 + H_sub8         gain_sub8 = what oracle relabelling of the sub-8 components WOULD recover
                                   (computed exactly from the saved prediction masks)

So the 27.0 % gap = H (heterogeneity, regions >= 8) + F (sub-8 components); F is split further into the part a
no-minimum-size rule could recover and the part that would still be heterogeneous.
Counterfactual ceilings: no minimum size = (gain + gain_sub8)/E; homogeneous regions = E_in/E.
"""
import json
import sys

import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1])); from mriseg_paths import P as _P, CODE as _CODE  # noqa: E402
sys.path.insert(0, str(_CODE / "revision_analysis"))
from rev_common import *  # noqa
from scipy import ndimage

d = load_dev()
cvc = d["class_voxel_counts"]; src = d["pred_class"]; n = d["n"]
E = 8_662_107; missed = 2_607_924                      # frozen totals of the original (tiled) analysis
if VARIANT == "slide":                                  # recompute totals from this variant's own masks
    E = missed = 0
    for f_ in range(5):
        for p_ in sorted((OUT / "dev" / f"fold{f_}" / "masks").glob("*.npz")):
            z_ = np.load(p_); pr_, sg_ = z_["pred"].astype(np.int64), z_["seg"].astype(np.int64)
            E += int(((pr_ != sg_) & (sg_ >= 0)).sum()); missed += int(((pr_ == 0) & (sg_ > 0)).sum())
valid = cvc.sum(1); u_src = cvc[np.arange(n), src]; u_max = cvc.max(1)
gain = float((u_max - u_src).sum()); E_in = float((valid - u_src).sum()); H = float((valid - u_max).sum())
N = E - missed
out = {"E_valid": E, "missed": missed, "non_missed": N, "gain_ceiling_numerator": gain,
       "ceiling_pct": 100 * gain / E, "E_in_regions_ge8": E_in, "H_heterogeneity_regions_ge8": H,
       "F_error_in_components_lt8": N - E_in}

# ---- exact sub-8 component analysis from saved masks
struct = ndimage.generate_binary_structure(3, 1)
gain_sub = 0; err_sub = 0; n_sub = 0; per_class_sub = {c: {"err": 0, "gain": 0} for c in range(1, 5)}
chk_missed = 0; chk_E = 0
topo = {"wrong_fg_class": 0, "over_segmentation": 0, "missed_tissue": 0}
for f in range(5):
    for npzp in sorted((OUT / "dev" / f"fold{f}" / "masks").glob("*.npz")):
        z = np.load(npzp); pred = z["pred"].astype(np.int64); seg = z["seg"].astype(np.int64)
        valid_vox = seg >= 0
        chk_E += int(((pred != seg) & valid_vox).sum()); chk_missed += int(((pred == 0) & (seg > 0)).sum())
        topo["wrong_fg_class"] += int(((pred > 0) & (seg > 0) & (pred != seg)).sum())
        topo["over_segmentation"] += int(((pred > 0) & (seg == 0)).sum())
        topo["missed_tissue"] += int(((pred == 0) & (seg > 0)).sum())
        for c in range(1, 5):
            m = pred == c
            if not m.any():
                continue
            lab, nc = ndimage.label(m, structure=struct)
            sizes = np.bincount(lab.ravel(), minlength=nc + 1)
            small = np.flatnonzero((sizes[1:] < 8) & (sizes[1:] > 0)) + 1
            if small.size == 0:
                continue
            objs = ndimage.find_objects(lab)
            for cid in small:
                sl = objs[cid - 1]; mm = lab[sl] == cid; sg = seg[sl][mm]
                counts = np.array([(sg == k).sum() for k in range(5)])
                err = counts.sum() - counts[c]; g = counts.max() - counts[c]
                err_sub += int(err); gain_sub += int(g); n_sub += 1
                per_class_sub[c]["err"] += int(err); per_class_sub[c]["gain"] += int(g)
out["topology_voxels"] = topo
out["topology_pct"] = {k: 100 * v / chk_E for k, v in topo.items()}
out["recomputed_total_error_from_masks"] = chk_E
out["recomputed_missed_from_masks"] = chk_missed
out["n_components_lt8"] = n_sub
out["F_error_in_components_lt8_from_masks"] = err_sub
out["gain_sub8_recoverable_by_oracle_relabel"] = gain_sub
out["H_sub8_residual"] = err_sub - gain_sub
out["per_class_sub8"] = {CLASS_NAMES[c]: v for c, v in per_class_sub.items()}
pct = lambda v: 100 * v / E
out["pct_of_total_error"] = {
    "missed_tissue (unreachable)": pct(missed),
    "ceiling: oracle relabel of regions >=8": pct(gain),
    "heterogeneity inside regions >=8 (H)": pct(H),
    "sub-8 components, recoverable by oracle relabel": pct(gain_sub),
    "sub-8 components, still heterogeneous": pct(err_sub - gain_sub),
}
out["counterfactual_ceilings_pct"] = {
    "as_published (min size 8, heterogeneous regions)": pct(gain),
    "no minimum size (heterogeneous regions)": pct(gain + gain_sub),
    "homogeneous regions, min size 8": pct(E_in),
    "no minimum size AND homogeneous (= all non-missed error)": pct(N),
}
gap = N - gain
out["gap_explained"] = {"gap_pct_of_error": pct(gap), "heterogeneity_share_of_gap": H / gap,
                         "sub8_share_of_gap": (N - E_in) / gap,
                         "of_which_sub8_recoverable_by_removing_min_size": gain_sub / gap}
(OUT / "A5_CEILING_DECOMPOSITION.json").write_text(json.dumps(out, indent=1))
print(json.dumps({k: out[k] for k in ("pct_of_total_error", "counterfactual_ceilings_pct", "gap_explained", "n_components_lt8")}, indent=1))
print("check E:", chk_E, "vs", E, "| missed:", chk_missed, "vs", missed)
