"""Main Figure 5: single evaluation on the 50 reserved patients. Data code unchanged; layout (2x2) and style redrawn."""
import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from figcommon import OUT, save  # noqa
from revstyle import *  # noqa

set_style()
HU = json.loads((OUT / "H_RESULTS_UTILITY.json").read_text())
HD = json.loads((OUT / "H_RESULTS_DICE_HD95.json").read_text())

fig, axs = plt.subplots(2, 2, figsize=(FIG_W, 5.9), gridspec_kw={"hspace": 0.62, "wspace": 0.38})

# (a) paired scatter, P2
ax = axs[0, 0]
pv = HU["P2"]["patient_values"]; T = np.array(pv["T"]); M = np.array(pv["M"])
lim = [min(T.min(), M.min()) - 0.01, max(T.max(), M.max()) + 0.01]
ax.plot(lim, lim, color=TAUPE, ls=":", lw=1.1); ax.axhline(0, color=TAUPE, lw=0.8); ax.axvline(0, color=TAUPE, lw=0.8)
ax.scatter(M, T, s=18, color=TEAL, alpha=0.75, edgecolor="white", linewidth=0.3, zorder=3)
d = HU["P2"]["paired_diff_T_minus_M"]
ax.set_xscale("symlog", linthresh=0.01); ax.set_yscale("symlog", linthresh=0.01)
ax.set_xlim(lim); ax.set_ylim(lim); ax.set_xlabel("Monolithic utility"); ax.set_ylabel("Typed utility")
for axis, setter in ((ax.xaxis, ax.set_xticks), (ax.yaxis, ax.set_yticks)):
    setter([-0.1, -0.01, 0, 0.01]); axis.set_ticklabels(["$-$0.1", "$-$0.01", "0", "0.01"])
    axis.set_minor_locator(matplotlib.ticker.NullLocator())
panel_title(ax, f"(a) P2, {len(T)} patients", f"difference {d[0]:+.4f} [{d[1]:+.4f}, {d[2]:+.4f}]")

# (b) mean utility by protocol
ax = axs[0, 1]
for i, p in enumerate(("P0", "P1", "P2")):
    for j, (k, col, mk) in enumerate((("T_mean_util_ci", TEAL, "o"), ("M_mean_util_ci", BRONZE, "s"))):
        m, lo, hi = HU[p][k]
        x = i + (j - 0.5) * 0.3
        ax.errorbar(x, m, yerr=[[m - lo], [hi - m]], fmt=mk, color=col, capsize=3, ms=6, lw=1.4,
                    label=("Typed" if j == 0 else "Monolithic") if i == 0 else None, zorder=3)
ax.axhline(0, color=CHARCOAL, lw=0.9); ax.set_ylim(-0.03, 0.01)
ax.set_xticks(range(3)); ax.set_xticklabels(["P0", "P1", "P2"]); ax.set_ylabel("Mean patient utility (95% CI)")
panel_title(ax, "(b) Utility by protocol"); ax.legend(fontsize=7.5, loc="lower left", handlelength=1.0, borderaxespad=0.2, labelspacing=0.25)

# (c) Dice change by class
ax = axs[1, 0]
CL = ["netc", "snfh", "et", "rc", "mean_foreground"]; CN = ["NETC", "SNFH", "ET", "RC", "Mean\nfg."]
for key, lab, col, mk, off in (("P2_typed", "Typed", TEAL, "o", -0.26), ("P2_mono", "Monolithic", BRONZE, "s", 0.0), ("oracle_relabel", "Oracle", CHARCOAL, "D", 0.26)):
    m = [HD["dice"][key][c]["delta_vs_baseline"][0] for c in CL]; lo = [HD["dice"][key][c]["delta_vs_baseline"][1] for c in CL]; hi = [HD["dice"][key][c]["delta_vs_baseline"][2] for c in CL]
    ax.errorbar(np.arange(5) + off, m, yerr=[np.array(m) - lo, np.array(hi) - m], fmt=mk, color=col, capsize=2.5, ms=5, lw=1.3, label=lab, zorder=3)
ax.axhline(0, color=CHARCOAL, lw=0.9); ax.set_xticks(range(5)); ax.set_xticklabels(CN, fontsize=7.5)
ax.set_ylabel(r"$\Delta$ Dice vs base"); panel_title(ax, "(c) Change in Dice"); ax.legend(fontsize=7.5, handlelength=1.0, borderaxespad=0.2, labelspacing=0.25)

# (d) per-network replicate differences, P2
ax = axs[1, 1]
pn = HU["P2"]["per_network"]
for i in range(5):
    m, lo, hi = pn[str(i)]["diff"]
    ax.errorbar(m, 4 - i, xerr=[[m - lo], [hi - m]], fmt="o", color=SLATE, capsize=2.5, ms=5.5, lw=1.4, zorder=3)
ax.axvline(0, color=CHARCOAL, lw=0.9); ax.set_yticks(range(5)); ax.set_yticklabels([f"Network {4 - k}" for k in range(5)], fontsize=7.5)
ax.grid(axis="y", visible=False)
ax.set_xlabel("Typed $-$ monolithic (95% CI)"); panel_title(ax, "(d) Base-network replicates")
save(fig, "New_reserved")
print("saved")
