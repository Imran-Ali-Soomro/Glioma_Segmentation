"""New main figure: corrected typed-vs-monolithic comparison and matched-coverage analysis (dev cohort)."""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1])); from mriseg_paths import P as _P, CODE as _CODE  # noqa: E402
sys.path.insert(0, str(_CODE / "revision_analysis"))
sys.path.insert(0, str(_CODE / "figures"))
from rev_common import *  # noqa
from ptdr_style import COLOR_MONO, COLOR_TYPED, COLOR_KEEP, set_style  # noqa

set_style()
sys.path.insert(0, str(Path(__file__).parent))
from figcommon import save  # noqa
d = load_dev(); src = d["pred_class"]
curves = json.loads((OUT / "A3_COVERAGE_CURVES.json").read_text())


def load(name):
    if name == "P0" and VARIANT != "slide":
        z = np.load(REPO / "PTDR_K6_OOF_PREDICTIONS.npz")
    elif name == "P0":
        z = np.load(OUT / "oof_P0_unw.npz")
    else:
        z = np.load(OUT / f"oof_{name}_unw.npz")
    return [np.mean([z[f"{a}_p_final_seed{s}"] for s in SEEDS], 0) for a in ("T", "M")]


fig = plt.figure(figsize=(10.4, 7.6))
gs = fig.add_gridspec(2, 6, hspace=0.42, wspace=1.9)
axA = fig.add_subplot(gs[0, 0:2]); axB = fig.add_subplot(gs[0, 2:4]); axC = fig.add_subplot(gs[0, 4:6])
axD = fig.add_subplot(gs[1, 0:3]); axE = fig.add_subplot(gs[1, 3:6])

# (a) paired patient utilities, P2
pT, pM = load("P2")
uT = patient_utility(delta_u(pT.argmax(1), d), d); uM = patient_utility(delta_u(pM.argmax(1), d), d)
lim = [min(uT.min(), uM.min()) - 0.02, max(uT.max(), uM.max()) + 0.02]
axA.plot(lim, lim, color="#999999", ls=":", lw=1); axA.axhline(0, color=COLOR_KEEP, lw=0.7); axA.axvline(0, color=COLOR_KEEP, lw=0.7)
axA.scatter(uM, uT, s=11, color="#444444", alpha=0.65, edgecolor="none")
m, lo, hi = boot_ci((uT - uM).values)
axA.set_xlim(lim); axA.set_ylim(lim); axA.set_xlabel("Monolithic (patient utility)"); axA.set_ylabel("Typed (patient utility)")
axA.set_title(f"(a) Nested protocol P2, 153 patients\npaired difference {m:+.4f} [{lo:+.4f}, {hi:+.4f}]", fontsize=8.5, loc="left")

# (b) utility-coverage curves, P2
c = curves["P2_unw"]; K = np.array(c["K"])
axB.plot(K, c["oracle_order_oracle_target"], color="#009E73", lw=1.2, ls="-.", label="Oracle ordering (ceiling)")
axB.plot(K, c["random_order_mono_target"], color="#999999", lw=1.0, ls=":", label="Random ordering")
axB.plot(K, c["typed"], color=COLOR_TYPED, lw=1.8, marker="o", ms=3, label="Typed")
axB.plot(K, c["mono"], color=COLOR_MONO, lw=1.4, ls="--", marker="s", ms=3, label="Monolithic")
axB.axhline(0, color="black", lw=0.6)
axB.axvline(c["typed_default_K"], color=COLOR_TYPED, lw=0.6, alpha=0.5); axB.axvline(c["mono_default_K"], color=COLOR_MONO, lw=0.6, alpha=0.5)
axB.set_ylim(-0.03, 0.05); axB.set_xlabel("Regions edited (coverage, top-K by P(change))"); axB.set_ylabel("Mean patient utility")
axB.set_title("(b) Utility versus coverage, P2", fontsize=8.5, loc="left"); axB.legend(fontsize=6.5, frameon=False, loc="center right")

# (c) utility-coverage, P0 (original protocol)
c0 = curves["P0_unw"]
axC.plot(K, c0["typed"], color=COLOR_TYPED, lw=1.8, marker="o", ms=3, label="Typed")
axC.plot(K, c0["mono"], color=COLOR_MONO, lw=1.4, ls="--", marker="s", ms=3, label="Monolithic")
axC.axhline(0, color="black", lw=0.6)
axC.axvline(c0["typed_default_K"], color=COLOR_TYPED, lw=0.6, alpha=0.5); axC.axvline(c0["mono_default_K"], color=COLOR_MONO, lw=0.6, alpha=0.5)
axC.set_ylim(-0.8, 0.06); axC.set_xlabel("Regions edited (coverage)"); axC.set_ylabel("Mean patient utility")
axC.set_title("(c) Utility versus coverage, original P0", fontsize=8.5, loc="left"); axC.legend(fontsize=6.5, frameon=False, loc="lower left")

# (d) mean utility by protocol with 95% CI
x0 = 0
labels = []
for i, (pname, lab) in enumerate((("P0", "P0 original\n(pooled, $S_2$)"), ("P1", "P1\n(pooled, $S_1$)"), ("P2", "P2 nested\n(per network, $S_2$)"))):
    pT_, pM_ = load(pname)
    for j, (p, col, mk) in enumerate(((pT_, COLOR_TYPED, "o"), (pM_, COLOR_MONO, "s"))):
        u = patient_utility(delta_u(p.argmax(1), d), d)
        mean, lo_, hi_ = boot_ci(u.values)
        xx = i + (j - 0.5) * 0.28
        axD.errorbar(xx, mean, yerr=[[mean - lo_], [hi_ - mean]], fmt=mk, color=col, capsize=3, ms=6, lw=1.2,
                     label=("Typed" if j == 0 else "Monolithic") if i == 0 else None)
        axD.text(xx + 0.05, hi_ + (0.002 if mean > -0.05 else 0.0) , f"{mean:+.3f}", ha="center", va="bottom", fontsize=7)
axD.axhline(0, color="black", lw=0.8); axD.text(0.3, 0.0006, "KEEP = 0", fontsize=7, ha="left", va="bottom")
axD.set_xticks(range(3)); axD.set_xticklabels(["P0 original\n(pooled, $S_2$)", "P1\n(pooled, $S_1$)", "P2 nested\n(per network, $S_2$)"], fontsize=8)
axD.set_yscale("symlog", linthresh=0.01); axD.set_ylim(-1.0, 0.03); axD.set_ylabel("Mean patient-level utility (95% CI)"); axD.legend(fontsize=7, frameon=False, loc="center right")
axD.set_title("(d) Mean utility by protocol", fontsize=8.5, loc="left")

# (e) paired difference T - M at default and matched coverage
a3 = json.loads((OUT / "A3_COMPARISON.json").read_text())
rows = []
for key, lab in (("P0_unw", "P0"), ("P1_unw", "P1"), ("P2_unw", "P2"), ("P2_size", "P2, size-weighted\nloss")):
    r = a3[key]
    rows.append((lab + " default", r["paired_diff_T_minus_M"]))
    rows.append((lab + " matched", r["matched_coverage"]["K=typed_default_coverage"]["paired_diff_T_minus_M"]))
for i, (lab, (m_, lo_, hi_)) in enumerate(rows[::-1]):
    col = COLOR_TYPED if "default" in lab else "#7F3C8D"
    axE.errorbar(m_, i, xerr=[[m_ - lo_], [hi_ - m_]], fmt="o" if "default" in lab else "D", color=col, capsize=2.5, ms=5, lw=1.1)
axE.axvline(0, color="black", lw=0.8)
axE.set_yticks(range(len(rows))); axE.set_yticklabels([r_[0].replace("\n", " ") for r_ in rows[::-1]], fontsize=7)
axE.set_xlabel("Paired difference, typed $-$ monolithic (95% CI)"); axE.set_xscale("symlog", linthresh=0.01)
axE.set_title("(e) Default versus matched coverage", fontsize=8.5, loc="left")
save(fig, "New_comparison")
print("saved")
