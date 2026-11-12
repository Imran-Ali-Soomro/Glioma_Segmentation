"""Train Model T / Model M under the revision protocols and save out-of-fold predictions.
P1 = S1 pooled cross-fit (valid cross-network features); P2 = per-network nested (S2).
weighting: 'unw' (the original unweighted NLL; primary) and 'size' (region-voxel-weighted NLL;
objective aligned with the voxel-weighted evaluation metric).
Also saves the patient/inner-fold assignment for release."""
import json
import sys
import time

import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1])); from mriseg_paths import P as _P, CODE as _CODE  # noqa: E402
sys.path.insert(0, str(_CODE / "revision_analysis"))
from rev_common import *  # noqa

d = load_dev()
inner, assign = p2_inner_assignment(d)
(OUT / "P2_PATIENT_INNER_FOLD_ASSIGNMENT.json").write_text(json.dumps(assign, indent=1, sort_keys=True))

jobs = [(p, w) for p in ("P2", "P1") for w in ("unw", "size")] + ([("P0", "unw")] if VARIANT == "slide" else [])
if len(sys.argv) > 1:                                    # optional job filter, e.g.  P2:unw P1:size
    jobs = [tuple(a.split(":")) for a in sys.argv[1:]]
for protocol, wname in jobs:
    path = OUT / f"oof_{protocol}_{wname}.npz"
    if path.exists():
        print("skip", path.name); continue
    weight = d["size_voxels"].astype(float) if wname == "size" else None
    save, t0 = {}, time.time()
    for seed in SEEDS:
        for arch in ("T", "M"):
            pf, pa = run_protocol(d, protocol, arch, seed, weight)
            save[f"{arch}_p_final_seed{seed}"] = pf
            if arch == "T":
                save[f"T_p_act_seed{seed}"] = pa
            print(f"{protocol}/{wname} seed={seed} arch={arch} done ({time.time()-t0:.0f}s)", flush=True)
    np.savez_compressed(path, **save)
    print("saved", path.name, flush=True)
print("ALL_DONE")
