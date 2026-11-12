"""Main Figure 1: (a) decision formulation, (b) evaluation design (nested per-network protocol).
Drawn at final print width in inches (1 data unit = 1 inch). Box heights follow the amount of text (no spare padding)."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, str(Path(__file__).parent))
from figcommon import save  # noqa
from revstyle import *  # noqa

set_style()
H = 5.6
fig = plt.figure(figsize=(FIG_W, H))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, FIG_W); ax.set_ylim(0, H); ax.axis("off"); ax.grid(False)

LEFT, RIGHT, W = 0.08, 2.84, 2.38
EC = "#8A7F68"
LINE = 0.123          # inches per 7.5 pt text line (with line spacing 1.18)
GAP = 0.11            # vertical gap between stacked boxes (holds the arrow)


def height(body, title=True):
    n = body.count("\n") + 1 if body else 0
    return (0.255 + LINE * n + 0.06) if body else 0.38


def box(x, ytop, w, title, body="", fc=BEIGE, tcolor=CHARCOAL, h=None):
    h = h or height(body)
    ax.add_patch(FancyBboxPatch((x, ytop - h), w, h, boxstyle="round,pad=0.0,rounding_size=0.06", fc=fc, ec=EC, lw=0.8, zorder=2))
    if body:
        ax.text(x + w / 2, ytop - 0.06, title, ha="center", va="top", fontsize=8, fontweight="bold", color=tcolor, zorder=3)
        ax.text(x + w / 2, ytop - 0.235, body, ha="center", va="top", fontsize=7.5, color=CHARCOAL, zorder=3, linespacing=1.18)
    else:
        ax.text(x + w / 2, ytop - h / 2, title, ha="center", va="center", fontsize=8, fontweight="bold", color=tcolor, zorder=3)
    return ytop - h


def arrow(x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=8, color=CHARCOAL, lw=0.9, zorder=1, shrinkA=0, shrinkB=0))


# ---------------- (a) decision formulation ----------------
ax.text(LEFT, H - 0.06, "(a) Decision formulation", fontsize=9, fontweight="bold", va="top")
cx = LEFT + W / 2
y = H - 0.32
stack = [("Four-modality MRI", "T1n  ·  T1c  ·  T2w  ·  FLAIR"),
         ("Frozen nnU-Net ResEnc-M", "five patient-grouped\nbase networks"),
         ("Predicted-class regions ($\\geq$8 voxels)", "class probabilities, source class,\nsize, pooled decoder features"),
         ("Region state", "11 features ($S_1$) or 75 ($S_2$)")]
for k, (t, b) in enumerate(stack):
    y1 = box(LEFT, y, W, t, b, TEAL_PALE)
    if k < len(stack) - 1:
        arrow(cx, y1, cx, y1 - GAP); y = y1 - GAP
    else:
        y = y1
typed = "7,142 parameters\naction head +\nsource-masked\ntransition head\n$P$(KEEP) = 1 $-$ $Q_{act}$\n$P(j)$ = $Q_{act}\\,Q_{tr}(j)$"
mono = "7,109 parameters\none five-class\nsoftmax, same\ntrunk, same final-\nlabel objective"
bw = 1.16
hm = height(typed)
yt = y - 0.2
arrow(LEFT + 0.55, y, LEFT + bw / 2, yt); arrow(LEFT + W - 0.55, y, LEFT + bw + 0.06 + bw / 2, yt)
box(LEFT, yt, bw, "Typed model", typed, TEAL_LIGHT, TEAL_DARK, h=hm)
box(LEFT + bw + 0.06, yt, bw, "Monolithic model", mono, BEIGE, "#7A4A22", h=hm)
yb = yt - hm - 0.15
arrow(LEFT + bw / 2, yt - hm, cx - 0.4, yb); arrow(LEFT + bw + 0.06 + bw / 2, yt - hm, cx + 0.4, yb)
box(LEFT, yb, W, "Final label per region", "BG · NETC · SNFH · ET · RC\n(KEEP: no change)", TEAL_PALE)

# ---------------- (b) evaluation design ----------------
ax.text(RIGHT, H - 0.06, "(b) Evaluation design", fontsize=9, fontweight="bold", va="top")
rcx = RIGHT + W / 2
hw = 1.16
x2 = RIGHT + hw + 0.06
y = H - 0.32
y1 = box(RIGHT, y, W, "MU-Glioma-Post: 203 patients", "", BEIGE_DARK, h=0.36)
y = y1 - 0.14
arrow(rcx - 0.3, y1, RIGHT + hw / 2, y); arrow(rcx + 0.3, y1, x2 + hw / 2, y)


def pair(y, left, right, hl=None):
    h = hl or max(height(left[1]), height(right[1]))
    box(RIGHT, y, hw, left[0], left[1], left[2], left[3], h=h)
    box(x2, y, hw, right[0], right[1], right[2], right[3], h=h)
    return y - h


def down(y, gap=GAP):
    arrow(RIGHT + hw / 2, y, RIGHT + hw / 2, y - gap); arrow(x2 + hw / 2, y, x2 + hw / 2, y - gap)
    return y - gap


T, R = TEAL_LIGHT, BEIGE
y = pair(y, ("Development", "153 patients\n439 cases,\nfive patient-\ngrouped folds", T, TEAL_DARK),
         ("Reserved", "50 patients\n140 cases,\nevaluated once\nat the end", R, "#7A4A22"))
y = down(y)
y = pair(y, ("Base network $f$", "trained on folds\n$\\neq f$; scores only\nfold-$f$ patients", TEAL_PALE, CHARCOAL),
         ("Five base networks", "each scores every\nreserved case\n(inference only)", TEAL_PALE, CHARCOAL))
y = down(y)
y = pair(y, ("Corrector (P2)", "per network, trained\nand tested only on\nfold-$f$ patients; none\nwas in network $f$'s\ntraining set", T, TEAL_DARK),
         ("Reserved eval.", "corrector $f$ applied\nto its regions;\npatient value = mean\nof 5 replicates", R, "#7A4A22"))
yn = y - 0.17
arrow(RIGHT + hw / 2, y, rcx - 0.3, yn); arrow(x2 + hw / 2, y, rcx + 0.3, yn)
y1 = box(RIGHT, yn, W, "Endpoints", "net utility (KEEP = 0)  ·  matched\ncoverage  ·  Dice and HD95", BEIGE_DARK)
y = y1 - 0.1
box(RIGHT, y, W, "Retained for comparison", "P0: pooled cross-fit, $S_2$ (original)\nP1: pooled cross-fit, $S_1$ only", BEIGE)

# crop the canvas to the content: the lowest box edge defines the bottom, so no empty band remains under the figure
ymin = min(p.get_y() for p in ax.patches if isinstance(p, FancyBboxPatch))
fig.set_size_inches(FIG_W, H - ymin + 0.03)
ax.set_ylim(ymin - 0.03, H)
save(fig, "New_framework", pad=0.02)
print("saved")
