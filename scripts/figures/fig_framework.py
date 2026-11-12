"""Revised Figure 1: (a) decision formulation, (b) revised evaluation design (nested per-network protocol)."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1])); from mriseg_paths import P as _P, CODE as _CODE  # noqa: E402
sys.path.insert(0, str(_CODE / "figures"))
from ptdr_style import set_style  # noqa

set_style()
FIG = _P(r"Paper1_Revision\DiscoverCV_Revision1_Overleaf\figures")
C = {"blue": "#E8F0F7", "orange": "#F9EDE4", "green": "#E6F0E6", "grey": "#EFEFEF", "pink": "#F5E6E6", "yellow": "#FBF0D5", "purple": "#EBE4F2"}
EC = "#6B7F94"


def box(ax, x, y, w, h, title, body="", fc="white"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.12", fc=fc, ec=EC, lw=0.9, zorder=2))
    if body:
        ax.text(x + w / 2, y + h * 0.66, title, ha="center", va="center", fontsize=8.2, fontweight="bold", zorder=3)
        ax.text(x + w / 2, y + h * 0.30, body, ha="center", va="center", fontsize=7.6, zorder=3, linespacing=1.15)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=8.2, fontweight="bold", zorder=3)


def arr(ax, p, q):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=9, color="#444444", lw=0.9, zorder=1))


fig, (A, B) = plt.subplots(1, 2, figsize=(10.4, 9.0), gridspec_kw={"width_ratios": [0.92, 1.1]})
for ax in (A, B):
    ax.set_xlim(0, 10); ax.set_ylim(0, 20); ax.axis("off")

A.text(0, 19.4, "(a) Region-level decision formulation", fontsize=10, fontweight="bold")
box(A, 1.2, 17.0, 7.6, 1.6, "Four-modality MRI", "T1n  ·  T1c  ·  T2w  ·  FLAIR", C["blue"])
box(A, 1.2, 14.0, 7.6, 1.9, "Frozen nnU-Net ResEnc-M", "five patient-grouped base networks", C["blue"])
box(A, 1.2, 10.9, 7.6, 2.2, "Predicted-class regions ($\\geq$8 voxels)", "mean class probabilities · source class · size\npooled decoder features of the same network", C["blue"])
box(A, 1.2, 8.2, 7.6, 1.7, "Region state", "11 features ($S_1$)  or  75 features ($S_2$)", C["blue"])
box(A, 0.1, 2.9, 4.7, 4.1, "Typed model", "7,142 parameters\naction head + masked\ntransition head\n$P$(KEEP) = 1 $-$ $Q_{act}$\n$P(j)$ = $Q_{act}\\times Q_{tr}(j)$", C["blue"])
box(A, 5.2, 2.9, 4.7, 4.1, "Monolithic model", "7,109 parameters\none five-class softmax\nsame input and trunk\nsame final-label\nobjective", C["orange"])
box(A, 1.2, 0.3, 7.6, 1.9, "Final label per region", "BG · NETC · SNFH · ET · RC (KEEP: no change)", C["blue"])
for y0, y1 in ((17.0, 15.9), (14.0, 13.1), (10.9, 9.9)):
    arr(A, (5, y0), (5, y1))
arr(A, (3.5, 8.2), (2.4, 7.0)); arr(A, (6.5, 8.2), (7.6, 7.0)); arr(A, (2.4, 2.9), (3.6, 2.2)); arr(A, (7.6, 2.9), (6.4, 2.2))

B.text(0, 19.4, "(b) Evaluation design", fontsize=10, fontweight="bold")
box(B, 2.2, 17.3, 5.6, 1.5, "MU-Glioma-Post", "203 patients", C["grey"])
box(B, 0.0, 14.6, 4.7, 1.9, "Development pool", "153 patients, 439 cases\n5 patient-grouped folds", C["blue"])
box(B, 5.3, 14.6, 4.7, 1.9, "Reserved cohort", "50 patients, 140 cases\nevaluated after all decisions were fixed", C["pink"])
arr(B, (4.2, 17.3), (2.4, 16.5)); arr(B, (5.8, 17.3), (7.6, 16.5))
box(B, 0.0, 11.4, 4.7, 2.3, "Base network $f$ (five)", "trained on the four folds $\\neq f$;\nscores only fold-$f$ patients", C["green"])
arr(B, (2.35, 14.6), (2.35, 13.7))
box(B, 0.0, 7.2, 4.7, 3.4, "Per-network corrector (P2)", "trained and tested only on\nfold-$f$ patients (inner 5-fold,\npatient-grouped): no corrector\npatient was in network $f$'s\nbase-model training set", C["green"])
arr(B, (2.35, 11.4), (2.35, 10.6))
box(B, 5.3, 7.2, 4.7, 3.4, "Reserved evaluation", "every base network scores all\nreserved cases; corrector $f$\n(trained on development fold $f$)\nis applied to its regions;\npatient value = mean of 5 replicates", C["pink"])
arr(B, (7.65, 14.6), (7.65, 10.6))
box(B, 5.3, 11.4, 4.7, 2.3, "Five base networks", "score every reserved case\n(inference only, frozen)", C["green"])
arr(B, (2.35, 14.6), (2.35, 13.7))
box(B, 1.0, 3.6, 8.0, 2.5, "Endpoints", "patient-level normalised net utility (KEEP = 0)\nmatched-coverage comparison  ·  Dice and HD95\nbefore and after correction", C["purple"])
arr(B, (2.35, 7.2), (3.6, 6.1)); arr(B, (7.65, 7.2), (6.4, 6.1))
box(B, 0.6, 0.4, 8.8, 2.4, "Retained for comparison", "P0: original two-stage pooled cross-fit ($S_2$); P1: pooled cross-fit, $S_1$ only.\nP2 replaces the non-nested design of the original submission.", C["yellow"])

fig.savefig(FIG / "Fig1.pdf", bbox_inches="tight")
fig.savefig(_P(r"Paper1_Revision\revision_analysis\outputs\New_framework.png"), dpi=130, bbox_inches="tight")
print("saved")
