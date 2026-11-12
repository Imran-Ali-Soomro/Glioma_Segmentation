"""New main figure: protocol audit of the action head (R1.5 / R2.5 / R1.6)."""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1])); from mriseg_paths import P as _P, CODE as _CODE  # noqa: E402
sys.path.insert(0, str(_CODE / "figures"))
from ptdr_style import COLOR_MONO, COLOR_TYPED, set_style  # noqa

set_style()
sys.path.insert(0, str(Path(__file__).parent))
from figcommon import OUT, save  # noqa
a1 = json.loads((OUT / "A1_PROTOCOL_AUDIT.json").read_text())
a3 = json.loads((OUT / "A3_COMPARISON.json").read_text())

fig = plt.figure(figsize=(10.2, 3.9))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.25], wspace=0.62)

for k, (fs, title) in enumerate((("S2", "(a) $S_2$: pooled decoder channels"), ("S1", "(b) $S_1$: probabilities, source class, size"))):
    ax = fig.add_subplot(gs[0, k])
    M = np.array(a1[f"transfer_matrix_{fs}"]["matrix_train_row_test_col"])
    im = ax.imshow(M, vmin=0.2, vmax=1.0, cmap="viridis")
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if M[i, j] < 0.62 else "black", fontweight="bold" if i == j else None)
    ax.set_xticks(range(5)); ax.set_yticks(range(5))
    ax.set_xlabel("Test base network"); ax.set_ylabel("Training base network")
    ax.set_title(title, fontsize=9, loc="left")
    for s in ax.spines.values():
        s.set_visible(False)
    if k == 1:
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04); cb.set_label("AUROC")

ax = fig.add_subplot(gs[0, 2])
p = a1["sklearn_mlp16_protocol_comparison"]; ab = a1["torch_ablation_S2"]; cmp3 = a3["P2_unw"]; cmp0 = a3["P0_unw"]
labels = ["$S_1$ MLP", "$S_2$ MLP", "Act-only head\n($S_2$)", "Joint Model T\n($S_2$)"]
cross = [p["S1_P0_cross_network"]["auroc"], p["S2_P0_cross_network"]["auroc"],
         ab["P0_act_only"]["auroc_pooled_over_seeds"], cmp0["T_act_auroc_pooled"]]
within = [p["S1_P2_within_network"]["auroc"], p["S2_P2_within_network"]["auroc"],
          ab["P2_act_only"]["auroc_pooled_over_seeds"], cmp3["T_act_auroc_pooled"]]
x = np.arange(4); w = 0.38
ax.bar(x - w / 2, cross, w, color=COLOR_MONO, label="Original protocol (P0): cross-network", edgecolor="black", linewidth=0.4)
ax.bar(x + w / 2, within, w, color=COLOR_TYPED, label="Nested per-network protocol (P2)", edgecolor="black", linewidth=0.4)
ax.axhline(0.5, color="#555555", ls=":", lw=1.0); ax.text(-0.45, 0.515, "chance", fontsize=7, ha="left", va="bottom")
for xi, (c, wv) in enumerate(zip(cross, within)):
    ax.text(xi - w / 2, c + 0.012, f"{c:.2f}", ha="center", fontsize=7); ax.text(xi + w / 2, wv + 0.012, f"{wv:.2f}", ha="center", fontsize=7)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7.5); ax.set_ylim(0, 1.32); ax.set_ylabel("AUROC (pooled out-of-fold)")
ax.set_title("(c) Action-head AUROC by protocol", fontsize=9, loc="left")
ax.legend(fontsize=6.8, loc="upper center", frameon=False, ncol=1)
save(fig, "New_protocol_audit")
print("saved")
