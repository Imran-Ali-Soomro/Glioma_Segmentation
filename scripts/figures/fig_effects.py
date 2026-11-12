"""Main Figure 4: RC mitigation (R2.6) and segmentation-level effect (R1.10 / R2.7). Data code unchanged; style redrawn."""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from figcommon import OUT, save  # noqa
from revstyle import *  # noqa

set_style()
a4 = json.loads((OUT / "A4_CALIBRATION_RC.json").read_text())
dsum = json.loads((OUT / "A7_DICE_HD95_SUMMARY_dev.json").read_text())

fig, axs = plt.subplots(2, 2, figsize=(FIG_W, 5.9), gridspec_kw={"hspace": 0.62, "wspace": 0.36})
ax = axs[0, 0]
for proto, ls, col, mk in (("P2", "-", TEAL, "o"), ("P0", "--", BRONZE, "s")):
    sw = a4[proto]["rc_specific_threshold_sweep_raw"]
    ts = sorted(sw, key=float)
    ax.plot(range(len(ts)), [sw[t]["rc_net"] for t in ts], ls, color=col, marker=mk, ms=4.5, lw=1.7,
            label=("Nested (P2)" if proto == "P2" else "Original (P0)"))
ax.set_xticks(range(len(ts))); ax.set_xticklabels([f"{float(t):.3g}" if float(t) <= 1 else "never" for t in ts], fontsize=7.5)
ax.axhline(0, color=CHARCOAL, lw=0.9)
ax.set_xlabel("RC authorization threshold"); ax.set_ylabel("Net RC edits\n(beneficial $-$ harmful)")
panel_title(ax, "(a) RC-specific threshold"); ax.legend(fontsize=7.5, loc="lower right", handlelength=1.8, borderaxespad=0.2)

ax = axs[0, 1]
for kind, col, mk, ls in (("raw", CHARCOAL, "o", "-"), ("platt", TEAL, "s", "--"), ("isotonic", OCHRE, "^", ":")):
    sw = a4["P2"]["global_threshold_sweep"][kind]
    ts = sorted(sw, key=float)
    ax.plot([float(t) for t in ts], [sw[t]["mean_util"] for t in ts], color=col, marker=mk, ms=4.5, lw=1.6, ls=ls,
            label=kind.capitalize() if kind != "raw" else "Uncalibrated")
ax.axhline(0, color=CHARCOAL, lw=0.9)
ax.set_xlabel(r"Global threshold on $Q_{act}$"); ax.set_ylabel("Mean patient utility")
panel_title(ax, "(b) Calibration, P2"); ax.legend(fontsize=7.5, loc="lower center", handlelength=1.8, borderaxespad=0.2)

CL = ["netc", "snfh", "et", "rc", "mean_foreground"]; CN = ["NETC", "SNFH", "ET", "RC", "Mean\nfg."]
for k, (metric, title, ylab) in enumerate((("dice", "(c) Change in Dice", r"$\Delta$ Dice vs base"),
                                           ("hd95", "(d) Change in HD95", r"$\Delta$ HD95 (mm) vs base"))):
    ax = axs[1, k]
    series = (("P2_typed", "Typed", TEAL, "o", -0.26), ("P2_mono", "Monolithic", BRONZE, "s", 0.0), ("oracle_relabel", "Oracle", CHARCOAL, "D", 0.26))
    for key, lab, col, mk, off in series:
        m = [dsum[metric][key][c]["delta_vs_baseline"][0] for c in CL]
        lo = [dsum[metric][key][c]["delta_vs_baseline"][1] for c in CL]; hi = [dsum[metric][key][c]["delta_vs_baseline"][2] for c in CL]
        ax.errorbar(np.arange(5) + off, m, yerr=[np.array(m) - lo, np.array(hi) - m], fmt=mk, color=col, capsize=2.5, ms=5, lw=1.3, label=lab, zorder=3)
    ax.axhline(0, color=CHARCOAL, lw=0.9); ax.set_xticks(range(5)); ax.set_xticklabels(CN, fontsize=7.5)
    ax.set_ylabel(ylab); panel_title(ax, title); ax.legend(fontsize=7.5, handlelength=1.0, borderaxespad=0.2, labelspacing=0.25)
save(fig, "New_effects")
print("saved")
