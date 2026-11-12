"""New main figure: single evaluation on the 50 reserved patients."""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from figcommon import OUT, save  # noqa
from ptdr_style import COLOR_MONO, COLOR_TYPED, COLOR_KEEP, set_style  # noqa

set_style()
HU = json.loads((OUT / "H_RESULTS_UTILITY.json").read_text())
HD = json.loads((OUT / "H_RESULTS_DICE_HD95.json").read_text())

fig, axs = plt.subplots(1, 4, figsize=(13.4, 3.7), gridspec_kw={"width_ratios": [1, 1, 1.25, 1.1], "wspace": 0.42})

# (a) paired scatter, P2
ax = axs[0]
pv = HU["P2"]["patient_values"]; T = np.array(pv["T"]); M = np.array(pv["M"])
lim = [min(T.min(), M.min()) - 0.01, max(T.max(), M.max()) + 0.01]
ax.plot(lim, lim, color="#999999", ls=":", lw=1); ax.axhline(0, color=COLOR_KEEP, lw=0.7); ax.axvline(0, color=COLOR_KEEP, lw=0.7)
ax.scatter(M, T, s=14, color="#444444", alpha=0.7, edgecolor="none")
d = HU["P2"]["paired_diff_T_minus_M"]
ax.set_xscale("symlog", linthresh=0.01); ax.set_yscale("symlog", linthresh=0.01)
ax.set_xlim(lim); ax.set_ylim(lim); ax.set_xlabel("Monolithic (patient utility)"); ax.set_ylabel("Typed (patient utility)")
ax.set_title(f"(a) P2, {len(T)} patients\ndifference {d[0]:+.4f} [{d[1]:+.4f}, {d[2]:+.4f}]", fontsize=8.5, loc="left")

# (b) mean utility by protocol
ax = axs[1]
for i, (p, lab) in enumerate((("P0", "P0"), ("P1", "P1"), ("P2", "P2"))):
    for j, (k, col, mk) in enumerate((("T_mean_util_ci", COLOR_TYPED, "o"), ("M_mean_util_ci", COLOR_MONO, "s"))):
        m, lo, hi = HU[p][k]
        x = i + (j - 0.5) * 0.28
        ax.errorbar(x, m, yerr=[[m - lo], [hi - m]], fmt=mk, color=col, capsize=3, ms=6, lw=1.2, label=("Typed" if j == 0 else "Monolithic") if i == 0 else None)
ax.axhline(0, color="black", lw=0.8); ax.set_ylim(-0.03, 0.01)
ax.set_xticks(range(3)); ax.set_xticklabels(["P0", "P1", "P2"]); ax.set_ylabel("Mean patient utility (95% CI)")
ax.set_title("(b) Utility by protocol", fontsize=8.5, loc="left"); ax.legend(fontsize=7, frameon=False, loc="lower left")

# (c) Dice change by class
ax = axs[2]
CL = ["netc", "snfh", "et", "rc", "mean_foreground"]; CN = ["NETC", "SNFH", "ET", "RC", "Mean\nfg."]
for key, lab, col, off in (("P2_typed", "Typed", COLOR_TYPED, -0.27), ("P2_mono", "Monolithic", COLOR_MONO, 0.0), ("oracle_relabel", "Oracle", "#009E73", 0.27)):
    m = [HD["dice"][key][c]["delta_vs_baseline"][0] for c in CL]; lo = [HD["dice"][key][c]["delta_vs_baseline"][1] for c in CL]; hi = [HD["dice"][key][c]["delta_vs_baseline"][2] for c in CL]
    ax.errorbar(np.arange(5) + off, m, yerr=[np.array(m) - lo, np.array(hi) - m], fmt="o", color=col, capsize=2.5, ms=5, lw=1.1, label=lab)
ax.axhline(0, color="black", lw=0.7); ax.set_xticks(range(5)); ax.set_xticklabels(CN, fontsize=8)
ax.set_ylabel(r"$\Delta$ Dice vs base segmenter"); ax.set_title("(c) Change in Dice", fontsize=8.5, loc="left"); ax.legend(fontsize=7, frameon=False)

# (d) per-network replicate differences, P2
ax = axs[3]
pn = HU["P2"]["per_network"]
for i in range(5):
    m, lo, hi = pn[str(i)]["diff"]
    ax.errorbar(m, 4 - i, xerr=[[m - lo], [hi - m]], fmt="o", color="#7F3C8D", capsize=2.5, ms=5, lw=1.1)
ax.axvline(0, color="black", lw=0.8); ax.set_yticks(range(5)); ax.set_yticklabels([f"Network {4 - k}" for k in range(5)], fontsize=8)
ax.set_xlabel("Typed $-$ monolithic (95% CI)"); ax.set_title("(d) Base-network replicates, P2", fontsize=8.5, loc="left")
save(fig, "New_reserved")
print("saved")
