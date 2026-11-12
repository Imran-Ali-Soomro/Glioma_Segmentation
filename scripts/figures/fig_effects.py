"""New main figure: RC mitigation (R2.6) and segmentation-level effect (R1.10 / R2.7)."""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from figcommon import OUT, save  # noqa
from ptdr_style import COLOR_MONO, COLOR_TYPED, COLOR_BENEFICIAL, COLOR_HARMFUL, set_style  # noqa

set_style()
a4 = json.loads((OUT / "A4_CALIBRATION_RC.json").read_text())
dsum = json.loads((OUT / "A7_DICE_HD95_SUMMARY_dev.json").read_text())

fig, axs = plt.subplots(2, 2, figsize=(10.2, 7.4))
ax = axs[0, 0]
for proto, ls, col in (("P2", "-", "#0072B2"), ("P0", "--", "#D55E00")):
    sw = a4[proto]["rc_specific_threshold_sweep_raw"]
    ts = sorted(sw, key=float)
    xs = [min(float(t), 1.0) for t in ts]
    ax.plot(range(len(ts)), [sw[t]["rc_net"] for t in ts], ls, color=col, marker="o", ms=4,
            label=("Nested protocol (P2)" if proto == "P2" else "Original protocol (P0)"))
ax.set_xticks(range(len(ts))); ax.set_xticklabels([f"{float(t):.3g}" if float(t) <= 1 else "never" for t in ts], fontsize=7.5)
ax.axhline(0, color="black", lw=0.7)
ax.set_xlabel(r"Authorization threshold for RC-target edits"); ax.set_ylabel("Net RC-target edits (beneficial $-$ harmful)")
ax.set_title("(a) RC-specific authorization threshold", fontsize=9, loc="left"); ax.legend(fontsize=7.5, frameon=False)

ax = axs[0, 1]
for kind, col, mk in (("raw", "#444444", "o"), ("platt", "#009E73", "s"), ("isotonic", "#CC79A7", "^")):
    sw = a4["P2"]["global_threshold_sweep"][kind]
    ts = sorted(sw, key=float)
    ax.plot([float(t) for t in ts], [sw[t]["mean_util"] for t in ts], color=col, marker=mk, ms=4, lw=1.2, label=kind.capitalize() if kind != "raw" else "Uncalibrated")
ax.axhline(0, color="black", lw=0.7)
ax.set_xlabel(r"Global authorization threshold on $Q_{act}$"); ax.set_ylabel("Mean patient-level utility")
ax.set_title("(b) Calibration and threshold, nested protocol P2", fontsize=9, loc="left"); ax.legend(fontsize=7.5, frameon=False, loc="lower center")

CL = ["netc", "snfh", "et", "rc", "mean_foreground"]; CN = ["NETC", "SNFH", "ET", "RC", "Mean\nforeground"]
for k, (metric, title, ylab) in enumerate((("dice", "(c) Change in Dice after correction", r"$\Delta$ Dice vs base segmenter"),
                                           ("hd95", "(d) Change in HD95 after correction", r"$\Delta$ HD95 (mm) vs base segmenter"))):
    ax = axs[1, k]
    series = (("P2_typed", "Typed", COLOR_TYPED, -0.27), ("P2_mono", "Monolithic", COLOR_MONO, 0.0), ("oracle_relabel", "Oracle relabelling", "#009E73", 0.27))
    for key, lab, col, off in series:
        m = [dsum[metric][key][c]["delta_vs_baseline"][0] for c in CL]
        lo = [dsum[metric][key][c]["delta_vs_baseline"][1] for c in CL]; hi = [dsum[metric][key][c]["delta_vs_baseline"][2] for c in CL]
        ax.errorbar(np.arange(5) + off, m, yerr=[np.array(m) - lo, np.array(hi) - m], fmt="o", color=col, capsize=2.5, ms=5, lw=1.1, label=lab)
    ax.axhline(0, color="black", lw=0.7); ax.set_xticks(range(5)); ax.set_xticklabels(CN, fontsize=8)
    ax.set_ylabel(ylab); ax.set_title(title, fontsize=9, loc="left"); ax.legend(fontsize=7.5, frameon=False)
fig.tight_layout()
save(fig, "New_effects")
print("saved")
