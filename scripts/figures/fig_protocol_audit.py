"""Main Figure 2: protocol audit of the action head (R1.5 / R2.5 / R1.6). Data code unchanged; layout and style redrawn."""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from figcommon import OUT, save  # noqa
from revstyle import *  # noqa

set_style()
a1 = json.loads((OUT / "A1_PROTOCOL_AUDIT.json").read_text())
a3 = json.loads((OUT / "A3_COMPARISON.json").read_text())

fig = plt.figure(figsize=(FIG_W, 5.2))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.95], hspace=0.5, wspace=0.34)

for k, (fs, title) in enumerate((("S2", "(a) $S_2$: decoder channels"), ("S1", "(b) $S_1$: probabilities, class, size"))):
    ax = fig.add_subplot(gs[0, k])
    M = np.array(a1[f"transfer_matrix_{fs}"]["matrix_train_row_test_col"])
    im = ax.imshow(M, vmin=0.2, vmax=1.0, cmap=HEAT)
    ax.grid(False)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7.5,
                    color="white" if M[i, j] > 0.77 else CHARCOAL, fontweight="bold" if i == j else "normal")
    ax.set_xticks(range(5)); ax.set_yticks(range(5))
    ax.set_xlabel("Test base network"); ax.set_ylabel("Training base network" if k == 0 else "")
    ax.set_title(title, fontsize=9, loc="left", pad=4)
    for s in ax.spines.values():
        s.set_visible(False)
    if k == 1:
        cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.04); cb.set_label("AUROC", fontsize=8); cb.ax.tick_params(labelsize=7.5)
        cb.outline.set_visible(False)

ax = fig.add_subplot(gs[1, :])
p = a1["sklearn_mlp16_protocol_comparison"]; ab = a1["torch_ablation_S2"]; cmp3 = a3["P2_unw"]; cmp0 = a3["P0_unw"]
labels = ["$S_1$ MLP", "$S_2$ MLP", "Action-only\nhead ($S_2$)", "Joint typed\nmodel ($S_2$)"]
cross = [p["S1_P0_cross_network"]["auroc"], p["S2_P0_cross_network"]["auroc"],
         ab["P0_act_only"]["auroc_pooled_over_seeds"], cmp0["T_act_auroc_pooled"]]
within = [p["S1_P2_within_network"]["auroc"], p["S2_P2_within_network"]["auroc"],
          ab["P2_act_only"]["auroc_pooled_over_seeds"], cmp3["T_act_auroc_pooled"]]
x = np.arange(4); w = 0.36
ax.bar(x - w / 2, cross, w, color=BRONZE, hatch="///", edgecolor="white", linewidth=0.0, label="Original protocol P0 (cross-network)")
ax.bar(x + w / 2, within, w, color=TEAL, edgecolor="white", linewidth=0.0, label="Nested protocol P2 (per network)")
ax.axhline(0.5, color=CHARCOAL, ls=(0, (4, 3)), lw=1.0)
ax.text(3.9, 0.515, "chance", fontsize=7.5, ha="right", va="bottom", color=CHARCOAL)
for xi, (c, wv) in enumerate(zip(cross, within)):
    ax.text(xi - w / 2, c + 0.015, f"{c:.2f}", ha="center", fontsize=7.5)
    ax.text(xi + w / 2, wv + 0.015, f"{wv:.2f}", ha="center", fontsize=7.5)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7.5); ax.set_ylim(0, 1.25); ax.set_xlim(-0.55, 3.92)
ax.set_ylabel("AUROC (pooled out-of-fold)"); ax.grid(axis="x", visible=False)
ax.set_title("(c) Action-head AUROC by protocol", fontsize=9, loc="left", pad=4)
ax.legend(fontsize=7.5, loc="upper left", ncol=2, columnspacing=1.0, handlelength=1.4, borderaxespad=0.1)
save(fig, "New_protocol_audit")
print("saved")
