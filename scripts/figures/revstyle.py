"""Shared style for the revision figures: beige / light-teal palette, readable text at final print size.

Figures are drawn at their final width (FIG_W inches, about the width of the journal text column) so that a
\\includegraphics[width=\\textwidth] scales them by about 0.9 only. Minimum text size is 7 pt before that scaling.
Series are separated by colour AND by marker / line style, so the figures survive greyscale printing.
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

FIG_W = 5.3                      # inches; journal text column is about 4.9 inches

# ---- palette: beige neutrals, teal primary, bronze secondary, charcoal for references --------------------------
BEIGE_BG = "#FBF8F1"             # axes background
BEIGE = "#F1E8D8"                # box fill
BEIGE_DARK = "#D9C8A9"           # box edge / light grid emphasis
GRID = "#E7DFCF"
TEAL = "#1F7F7B"                 # typed
TEAL_DARK = "#14585A"
TEAL_LIGHT = "#BFE2DC"           # light teal fill
TEAL_PALE = "#E4F3F0"
BRONZE = "#A9672F"               # monolithic
OCHRE = "#CC9A3A"
CHARCOAL = "#3D4442"             # reference / oracle / text
TAUPE = "#8C8576"                # KEEP / chance lines
SLATE = "#5C7C86"                # extra series (blue-grey, near teal family but darker and greyer)

COLOR_TYPED, COLOR_MONO = TEAL, BRONZE
COLOR_KEEP, COLOR_ORACLE = TAUPE, CHARCOAL
MARK_T, MARK_M = "o", "s"
HEAT = LinearSegmentedColormap.from_list("beige_teal", ["#F6EFE0", "#CFE6DD", "#7DBDB4", "#2F8F8B", "#14585A"])


def set_style():
    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8.5,
        "axes.titlesize": 9,
        "axes.titleweight": "bold",
        "axes.labelsize": 8.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7.5,
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.edgecolor": "#6F6A5E",
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "axes.labelcolor": CHARCOAL,
        "text.color": CHARCOAL,
        "xtick.color": CHARCOAL,
        "ytick.color": CHARCOAL,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "lines.linewidth": 1.6,
        "lines.markersize": 5.5,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })


def panel_title(ax, text, sub=None):
    """Bold panel title, left aligned; optional regular-weight second line."""
    ax.set_title(text, loc="left", fontsize=9, pad=4 if sub is None else 15)
    if sub is not None:
        ax.text(0, 1.02, sub, transform=ax.transAxes, fontsize=7.5, fontweight="normal", va="bottom", ha="left", color=CHARCOAL)


def zero_line(ax, **kw):
    ax.axhline(0, color=CHARCOAL, lw=0.9, zorder=1, **kw)
