"""Patient-level summary of per-case Dice / HD95 / volume tables (development and reserved cohorts)."""
import numpy as np
import pandas as pd

from rev_common import boot_ci

CLASSES = ["netc", "snfh", "et", "rc"]


def _patient_table(df, col):
    """patient x (variant,class) table: mean over that patient's cases (and over base-network
    replicates, if a 'network' column exists), ignoring undefined (NaN) entries."""
    g = df.groupby(["patient", "variant", "cls"])[col].mean().unstack(["variant", "cls"])
    return g


def summarize(df, variants, base="baseline"):
    out = {}
    for col in ("dice", "hd95"):
        t = _patient_table(df, col)
        out[col] = {}
        for v in variants:
            cls_res = {}
            for c in CLASSES:
                x = t[(v, c)].dropna()
                cls_res[c] = {"mean": float(x.mean()), "n_patients": int(len(x))}
                if v != base:
                    paired = (t[(v, c)] - t[(base, c)]).dropna()
                    m, lo, hi = boot_ci(paired.values) if len(paired) > 1 else (np.nan, np.nan, np.nan)
                    cls_res[c]["delta_vs_baseline"] = [m, lo, hi]
            # mean foreground (patient-level mean over the four classes, ignoring undefined)
            fg = pd.concat([t[(v, c)] for c in CLASSES], axis=1).mean(axis=1).dropna()
            cls_res["mean_foreground"] = {"mean": float(fg.mean()), "n_patients": int(len(fg))}
            if v != base:
                fgb = pd.concat([t[(base, c)] for c in CLASSES], axis=1).mean(axis=1)
                paired = (fg - fgb.loc[fg.index]).dropna()
                cls_res["mean_foreground"]["delta_vs_baseline"] = list(boot_ci(paired.values))
            out[col][v] = cls_res
    # paired typed-minus-monolithic contrasts
    pairs = [("P2_typed", "P2_mono"), ("P0_typed", "P0_mono"), ("P2_typed", "P2_mono_matchedK")]
    out["typed_minus_mono"] = {}
    for a, b in pairs:
        if a not in variants or b not in variants:
            continue
        for col in ("dice", "hd95"):
            t = _patient_table(df, col)
            res = {}
            for c in CLASSES:
                p = (t[(a, c)] - t[(b, c)]).dropna()
                res[c] = list(boot_ci(p.values)) if len(p) > 1 else None
            fa = pd.concat([t[(a, c)] for c in CLASSES], axis=1).mean(axis=1)
            fb = pd.concat([t[(b, c)] for c in CLASSES], axis=1).mean(axis=1)
            res["mean_foreground"] = list(boot_ci((fa - fb).dropna().values))
            out["typed_minus_mono"][f"{a}_minus_{b}__{col}"] = res
    # volume error (signed, predicted - reference), patient-level, per class
    d2 = df.copy(); d2["vol_err"] = d2["vol_pred"] - d2["vol_gt"]
    d2["abs_vol_err"] = d2["vol_err"].abs()
    out["volume_error_ml"] = {}
    for col in ("vol_err", "abs_vol_err"):
        t = d2.groupby(["patient", "variant", "cls"])[col].mean().unstack(["variant", "cls"]) / 1000.0
        out["volume_error_ml"][col] = {v: {c: float(t[(v, c)].mean()) for c in CLASSES} for v in variants}
    return out
