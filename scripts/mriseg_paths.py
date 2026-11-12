"""Path configuration for the release (no machine-specific path appears anywhere else).

PKG        release root (folder that contains scripts/, derived/, protocol/, splits/, manuscript/)
CODE       PKG/scripts
DATA_ROOT  original project tree (data/, nnUNet_data/, split_v2/ ...); only needed to retrain or re-extract.
           Override with the environment variable MRISEG_ROOT.
WORK       output location; override with MRISEG_WORK.
P(rel)     maps a path that was relative to the authors' project root to the corresponding location here.
"""
import os
import re
from pathlib import Path

PKG = Path(os.environ.get("MRISEG_PACKAGE", Path(__file__).resolve().parents[1]))
CODE = PKG / "scripts"
DATA_ROOT = Path(os.environ.get("MRISEG_ROOT", PKG))
WORK = Path(os.environ.get("MRISEG_WORK", PKG / "work"))


def P(rel=""):
    parts = [x for x in re.split(r"[\\/]+", str(rel)) if x]
    if parts[:2] == ["Paper1_Revision", "revision_analysis"]:
        base, parts = WORK / "revision_analysis", parts[2:]
    elif parts[:2] == ["Paper1_Revision", "DiscoverCV_Revision1_Overleaf"]:
        base, parts = WORK / "manuscript", parts[2:]
    elif parts[:1] == ["Paper1_Revision"]:
        base, parts = WORK / "submission", parts[1:]
    elif parts[:2] == ["figures", "scripts"]:
        base, parts = CODE / "figures", parts[2:]
    else:
        base = DATA_ROOT
    return base.joinpath(*parts)
