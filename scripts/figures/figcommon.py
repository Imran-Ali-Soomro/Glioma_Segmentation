"""Shared paths for the revision figures. REV_VARIANT=slide (default, primary) or tiled (as-submitted pipeline)."""
import os
import sys
from pathlib import Path

import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1])); from mriseg_paths import P as _P, CODE as _CODE  # noqa: E402
sys.path.insert(0, str(_CODE / "figures"))
sys.path.insert(0, str(_CODE / "revision_analysis"))
VARIANT = os.environ.get("REV_VARIANT", "slide")
REV = _P(r"Paper1_Revision\revision_analysis")
OUT = REV / ("outputs_slide" if VARIANT == "slide" else "outputs")
FIGDIR = _P(r"Paper1_Revision\DiscoverCV_Revision1_Overleaf\figures")
SUF = "" if VARIANT == "slide" else "_tiled"


FINAL = {"New_framework": "Fig1", "New_protocol_audit": "Fig2", "New_comparison": "Fig3", "New_effects": "Fig4", "New_reserved": "Fig5"}


def save(fig, name):
    fig.savefig(FIGDIR / f"{FINAL.get(name, name)}{SUF}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}{SUF}.png", dpi=150, bbox_inches="tight")
