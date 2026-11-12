"""Main Figure 3: corrected typed-vs-monolithic comparison and matched-coverage analysis (development cohort).
Data code unchanged; layout, colours and text sizes redrawn for the final print width."""
import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1])); from mriseg_paths import P as _P, CODE as _CODE  # noqa: E402
sys.path.insert(0, str(_CODE / "revision_analysis"))
from rev_common import *  # noqa

sys.path.insert(0, str(Path(__file__).parent))
from figcommon import save  # noqa
from revstyle import *  # noqa

set_style()
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


fig = plt.figure(figsize=(FIG_W, 6.4))
gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.95], hspace=0.66, wspace=0.36, left=0.12, right=0.98, top=0.95, bottom=0.08)
axA, axB = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
axC, axD = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
_pos = gs[2, :].get_position(fig)
axE = fig.add_axes([0.2, _pos.y0, 0.98 - 0.2, _pos.height])          # group labels are short, so panel (e) can use almost the full width

# (a) paired patient utilities, P2
pT, pM = load("P2")
uT = patient_utility(delta_u(pT.argmax(1), d), d); uM = patient_utility(delta_u(pM.argmax(1), d), d)
lim = [min(uT.min(), uM.min()) - 0.02, max(uT.max(), uM.max()) + 0.02]
axA.plot(lim, lim, color=TAUPE, ls=":", lw=1.1)
axA.axhline(0, color=TAUPE, lw=0.8); axA.axvline(0, color=TAUPE, lw=0.8)
axA.scatter(uM, uT, s=16, color=TEAL, alpha=0.7, edgecolor="white", linewidth=0.3, zorder=3)
m, lo, hi = boot_ci((uT - uM).values)
axA.set_xlim(lim); axA.set_ylim(lim); axA.set_xlabel("Monolithic utility"); axA.set_ylabel("Typed utility")
panel_title(axA, "(a) Paired patients, P2", f"difference {m:+.4f} [{lo:+.4f}, {hi:+.4f}]")

# (b) utility-coverage curves, P2
c = curves["P2_unw"]; K = np.array(c["K"])
axB.plot(K, c["oracle_order_oracle_target"], color=CHARCOAL, lw=1.3, ls="-.", label="Oracle ordering")
axB.plot(K, c["random_order_mono_target"], color=TAUPE, lw=1.2, ls=":", label="Random ordering")
axB.plot(K, c["typed"], color=TEAL, lw=1.9, marker="o", ms=3.5, label="Typed")
axB.plot(K, c["mono"], color=BRONZE, lw=1.6, ls="--", marker="s", ms=3.5, label="Monolithic")
axB.axhline(0, color=CHARCOAL, lw=0.8)
axB.axvline(c["typed_default_K"], color=TEAL, lw=0.8, alpha=0.55); axB.axvline(c["mono_default_K"], color=BRONZE, lw=0.8, alpha=0.55)
axB.set_ylim(-0.03, 0.05); axB.set_xlabel("Regions edited (top-$K$)"); axB.set_ylabel("Mean patient utility")
panel_title(axB, "(b) Utility vs coverage, P2"); axB.legend(fontsize=7, loc="upper right", bbox_to_anchor=(1.0, 0.8), ncol=2, columnspacing=0.8, handlelength=1.6, borderaxespad=0.1, labelspacing=0.25)

# (c) utility-coverage, P0 (original protocol)
c0 = curves["P0_unw"]
axC.plot(K, c0["typed"], color=TEAL, lw=1.9, marker="o", ms=3.5, label="Typed")
axC.plot(K, c0["mono"], color=BRONZE, lw=1.6, ls="--", marker="s", ms=3.5, label="Monolithic")
axC.axhline(0, color=CHARCOAL, lw=0.8)
axC.axvline(c0["typed_default_K"], color=TEAL, lw=0.8, alpha=0.55); axC.axvline(c0["mono_default_K"], color=BRONZE, lw=0.8, alpha=0.55)
axC.set_ylim(-0.8, 0.06); axC.set_xlabel("Regions edited (top-$K$)"); axC.set_ylabel("Mean patient utility")
panel_title(axC, "(c) Utility vs coverage, P0"); axC.legend(fontsize=7, loc="lower left", handlelength=1.8, borderaxespad=0.2, labelspacing=0.25)

# (d) mean utility by protocol with 95% CI
for i, pname in enumerate(("P0", "P1", "P2")):
    pT_, pM_ = load(pname)
    for j, (p, col, mk) in enumerate(((pT_, TEAL, "o"), (pM_, BRONZE, "s"))):
        u = patient_utility(delta_u(p.argmax(1), d), d)
        mean, lo_, hi_ = boot_ci(u.values)
        xx = i + (j - 0.5) * 0.34
        axD.errorbar(xx, mean, yerr=[[mean - lo_], [hi_ - mean]], fmt=mk, color=col, capsize=3, ms=6, lw=1.4,
                     label=("Typed" if j == 0 else "Monolithic") if i == 0 else None, zorder=3)
        if pname == "P0":      # P1 and P2 means (all within +-0.001 of KEEP) are in Table 3; labels there would overlap
            axD.text(xx + 0.2, mean, f"{mean:+.3f}", ha="left", va="center", fontsize=7.5)
axD.axhline(0, color=CHARCOAL, lw=0.9)
axD.set_xticks(range(3)); axD.set_xticklabels(["P0\n(pooled $S_2$)", "P1\n(pooled $S_1$)", "P2\n(nested $S_2$)"], fontsize=7.5)
axD.set_xlim(-0.6, 2.7)
axD.set_yscale("symlog", linthresh=0.01); axD.set_ylim(-1.0, 0.03); axD.set_ylabel("Mean patient utility (95% CI)")
axD.set_yticks([-1, -0.1, -0.01, 0, 0.01]); axD.set_yticklabels(["$-$1", "$-$0.1", "$-$0.01", "0", "0.01"])
axD.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
axD.legend(fontsize=7, loc="center left", handlelength=1.2, borderaxespad=0.2, labelspacing=0.25)
panel_title(axD, "(d) Mean utility by protocol")

# (e) paired difference T - M at default and matched coverage
a3 = json.loads((OUT / "A3_COMPARISON.json").read_text())
rows = []
for key, lab in (("P0_unw", "P0"), ("P1_unw", "P1"), ("P2_unw", "P2"), ("P2_size", "P2 size-weighted")):
    r = a3[key]
    rows.append((lab + ", default", r["paired_diff_T_minus_M"]))
    rows.append((lab + ", matched", r["matched_coverage"]["K=typed_default_coverage"]["paired_diff_T_minus_M"]))
for i, (lab, (m_, lo_, hi_)) in enumerate(rows[::-1]):
    default = "default" in lab
    axE.errorbar(m_, i, xerr=[[m_ - lo_], [hi_ - m_]], fmt="o" if default else "D", color=TEAL if default else CHARCOAL,
                 capsize=3, ms=5.5, lw=1.4, zorder=3)
axE.axvline(0, color=CHARCOAL, lw=0.9)
axE.set_yticks([6.5, 4.5, 2.5, 0.5]); axE.set_yticklabels(["P0", "P1", "P2", "P2 size-\nweighted"], fontsize=7.5)
axE.set_ylim(-0.7, 7.7)
for _y in (1.5, 3.5, 5.5):
    axE.axhline(_y, color=GRID, lw=0.8, zorder=0)       # separates the four protocol groups
axE.set_xlabel("Paired difference, typed $-$ monolithic (95% CI)"); axE.set_xscale("symlog", linthresh=0.01)
axE.set_xlim(-0.03, 0.9)
axE.set_xticks([-0.01, 0, 0.01, 0.1, 0.5]); axE.set_xticklabels(["$-$0.01", "0", "0.01", "0.1", "0.5"])
axE.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
axE.grid(axis="y", visible=False)
panel_title(axE, "(e) Default versus matched coverage")
from matplotlib.lines import Line2D
axE.legend(handles=[Line2D([], [], marker="o", color=TEAL, lw=0, ms=5.5, label="Default coverage"),
                    Line2D([], [], marker="D", color=CHARCOAL, lw=0, ms=5, label="Matched coverage")],
           fontsize=7.5, loc="lower right", borderaxespad=0.2)
save(fig, "New_comparison")
print("saved")
