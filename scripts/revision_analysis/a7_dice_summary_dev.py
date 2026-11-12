import json, sys
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1])); from mriseg_paths import P as _P, CODE as _CODE  # noqa: E402
sys.path.insert(0, str(_CODE / "revision_analysis"))
from rev_common import *  # noqa
from dice_summary import summarize
df = pd.read_csv(OUT / "A6_DICE_HD95_dev.csv")
variants = ["baseline", "P0_typed", "P0_mono", "P2_typed", "P2_mono", "P2_mono_matchedK", "oracle_relabel"]
s = summarize(df, variants)
(OUT / "A7_DICE_HD95_SUMMARY_dev.json").write_text(json.dumps(s, indent=1, default=float))
f = lambda x: [round(v, 4) for v in x]
for col in ("dice", "hd95"):
    print("=====", col)
    for v in variants:
        r = s[col][v]
        print(f"{v:18s}", {c: round(r[c]["mean"], 4) for c in ("netc", "snfh", "et", "rc", "mean_foreground")})
    for v in ("P2_typed", "P2_mono", "P0_typed", "P0_mono", "oracle_relabel"):
        r = s[col][v]
        print(f"  delta vs baseline {v}:", {c: f(r[c]["delta_vs_baseline"]) for c in ("netc", "snfh", "et", "rc", "mean_foreground")})
print("typed - mono:", json.dumps({k: {c: (f(x) if x else None) for c, x in v.items()} for k, v in s["typed_minus_mono"].items()}, indent=0))
print("vol err (ml) signed:", s["volume_error_ml"]["vol_err"]["baseline"], s["volume_error_ml"]["vol_err"]["P2_typed"])
print("vol err (ml) abs:", s["volume_error_ml"]["abs_vol_err"]["baseline"], s["volume_error_ml"]["abs_vol_err"]["P2_typed"])
