"""Shared plotting style/utilities for the PTDR publication figure package.

Font: Times New Roman was confirmed available via matplotlib.font_manager on this
Windows environment (checked explicitly before committing to it -- see
figures/PTDR_FIGURE_SOURCE_LEDGER.md). Liberation Serif is configured as the
documented fallback if Times New Roman is ever unavailable in a different
environment that reuses these scripts.

Color encoding is FIXED and IDENTICAL across every figure in this package:
  TYPED        -- Model T / typed-factorized decisions
  MONO         -- Model M / monolithic final-action classifier
  KEEP_BASE    -- KEEP / base-segmentation-unchanged outcome
  BENEFICIAL   -- a beneficial (utility-improving) edit
  HARMFUL      -- a harmful (utility-reducing) edit
Each is additionally distinguished by marker/linestyle, not color alone.
Palette is the Okabe-Ito colorblind-safe set (Okabe & Ito, 2008).
"""
import json
from pathlib import Path

import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1])); from mriseg_paths import P as _P, CODE as _CODE  # noqa: E402
REPO_ROOT = _P(r"")
FIG_ROOT = REPO_ROOT / "figures"
MAIN_DIR = FIG_ROOT / "main"
SUPP_DIR = FIG_ROOT / "supplement"
SRC_DATA_DIR = FIG_ROOT / "source_data"

CLASS_KEYS = ["bg", "netc", "snfh", "et", "rc"]
CLASS_LABELS = ["BG", "NETC", "SNFH", "ET", "RC"]

# ---- fixed semantic color/marker/linestyle encoding (Okabe-Ito palette) ----
COLOR_TYPED = "#0072B2"       # blue
COLOR_MONO = "#D55E00"        # vermillion
COLOR_KEEP = "#999999"        # neutral gray
COLOR_BENEFICIAL = "#009E73"  # bluish green
COLOR_HARMFUL = "#B2182B"     # dark red (distinct hue from vermillion)
COLOR_NEUTRAL_GRID = "#CCCCCC"

MARKER_TYPED = "o"
MARKER_MONO = "s"
MARKER_BENEFICIAL = "^"
MARKER_HARMFUL = "x"

LS_TYPED = "-"
LS_MONO = "--"

# per-class consistent qualitative palette (used only where classes, not
# typed/mono/beneficial/harmful, are being distinguished) -- Okabe-Ito again
CLASS_COLORS = {
    "bg": "#999999", "netc": "#0072B2", "snfh": "#009E73",
    "et": "#E69F00", "rc": "#B2182B",
}


def _times_new_roman_available():
    names = set(f.name for f in fm.fontManager.ttflist)
    return "Times New Roman" in names


FONT_USED = "Times New Roman" if _times_new_roman_available() else "Liberation Serif"


def set_style():
    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.serif": [FONT_USED, "Liberation Serif", "DejaVu Serif"],
        # matplotlib's mathtext ($...$) is NOT governed by font.family/font.serif --
        # its default fontset ("dejavusans") silently renders all math-mode text in
        # DejaVu Sans regardless of the regular-text font setting above, which is the
        # confirmed root cause of the DejaVu Sans contamination found in Figures 1 and
        # 5 (the only two figures containing $...$ equations). "stix" is a Times-like
        # serif mathtext set, matching the regular Times New Roman text.
        "mathtext.fontset": "stix",
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.titlesize": 9,
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.edgecolor": "black",
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })


set_style()


def save_main_figure(fig, name):
    """Main figures: SVG + PDF + TIFF at 600 dpi, per the governing task."""
    for ext in ["svg", "pdf", "tiff"]:
        path = MAIN_DIR / f"{name}.{ext}"
        fig.savefig(path, dpi=600, bbox_inches="tight")
    print(f"Saved main figure: {name} (svg, pdf, tiff @ 600dpi)")


def save_supplement_figure(fig, name):
    """Supplementary figures: PDF + PNG at 600 dpi (documented choice, see
    PTDR_FIGURE_QA.md -- main figures use the full svg/pdf/tiff triplet per
    the governing task; supplements use a lighter but still publication-grade
    pairing)."""
    for ext in ["pdf", "png"]:
        path = SUPP_DIR / f"{name}.{ext}"
        fig.savefig(path, dpi=600, bbox_inches="tight")
    print(f"Saved supplementary figure: {name} (pdf, png @ 600dpi)")


def save_source_data(df, name):
    path = SRC_DATA_DIR / f"{name}_sourcedata.csv"
    df.to_csv(path, index=False)
    print(f"Saved source data: {path}")


def load_json(relpath):
    with open(REPO_ROOT / relpath) as f:
        return json.load(f)
