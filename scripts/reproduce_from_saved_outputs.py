"""Minimal end-to-end reproduction ENTRYPOINT (no GPU, no MU-Glioma-Post images, no retraining).

What it reproduces: every number in the manuscript text and in the generated supplementary tables, from the saved
result files in derived/results.  It stages those files, renders manuscript/main_template.tex into main.tex and builds
supp_tables_generated.tex, then compares both outputs byte-for-byte with the versions shipped in manuscript/reference/.

What it does NOT do: retrain the base networks, re-run inference, or retrain the correctors.  Those steps (see README,
"Reproduction from raw data") need the MU-Glioma-Post images and a GPU, and are outside this entrypoint.

    python scripts/reproduce_from_saved_outputs.py            # from the release root
Exit code 0 = outputs identical to the shipped reference; 1 = difference.
"""
import difflib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mriseg_paths import PKG, WORK  # noqa: E402

REV = WORK / "revision_analysis"
MAN = WORK / "manuscript"
RES = PKG / "derived" / "results"
for d in (REV / "outputs_slide", REV / "outputs", MAN):
    d.mkdir(parents=True, exist_ok=True)

n = 0
for p in sorted(RES.iterdir()):
    for prefix, dest in (("slide_", "outputs_slide"), ("tiled_", "outputs")):
        if p.name.startswith(prefix):
            shutil.copy2(p, REV / dest / p.name[len(prefix):]); n += 1
shutil.copy2(PKG / "protocol" / "static_numbers.json", REV / "static_numbers.json")
print(f"staged {n} result files into {REV}")

TEMPLATE = PKG / "manuscript" / "main_template.tex"
if not TEMPLATE.exists():
    # The manuscript text is not distributed with this release: verify the saved results and print the headline numbers.
    A3 = json.loads((REV / "outputs_slide" / "A3_COMPARISON.json").read_text())["P2_unw"]
    HU = json.loads((REV / "outputs_slide" / "H_RESULTS_UTILITY.json").read_text())["P2"]
    d, h = A3["paired_diff_T_minus_M"], HU["paired_diff_T_minus_M"]
    print("Manuscript text not distributed: printing the headline numbers from the saved result files.")
    print("Development cohort, primary protocol P2 (KEEP = 0):")
    print(f"  typed {A3['T_mean_util']:+.4f}, monolithic {A3['M_mean_util']:+.4f}, typed - monolithic {d[0]:+.4f} (95% CI {d[1]:+.4f} to {d[2]:+.4f})")
    print("Reserved cohort, P2:")
    print(f"  typed {HU['T_mean_util']:+.4f}, monolithic {HU['M_mean_util']:+.4f}, typed - monolithic {h[0]:+.4f} (95% CI {h[1]:+.4f} to {h[2]:+.4f})")
    sys.exit(0)
shutil.copy2(TEMPLATE, REV / "main_template.tex")

env = dict(os.environ, KMP_DUPLICATE_LIB_OK="TRUE", REV_VARIANT="slide", PYTHONIOENCODING="utf-8")
code = Path(__file__).resolve().parent / "revision_analysis"
for script, args in (("render_manuscript.py", ["main_template.tex"]), ("make_supp_tables.py", [])):
    r = subprocess.run([sys.executable, str(code / script), *args], env=env, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout, r.stderr); sys.exit(f"{script} failed")
    print(r.stdout.strip().splitlines()[-1] if r.stdout.strip() else f"{script} ok")

ok = True
for name in ("main.tex", "supp_tables_generated.tex"):
    new = (MAN / name).read_text(encoding="utf-8")
    ref = (PKG / "manuscript" / "reference" / name).read_text(encoding="utf-8")
    if new == ref:
        print(f"MATCH     {name}  ({len(new):,} characters)")
    else:
        ok = False
        diff = list(difflib.unified_diff(ref.splitlines(), new.splitlines(), "reference", "reproduced", lineterm="", n=0))
        print(f"DIFFERENT {name}: {len(diff)} diff lines; first lines:"); print("\n".join(diff[:10]))

A3 = json.loads((REV / "outputs_slide" / "A3_COMPARISON.json").read_text())["P2_unw"]
print("\nHeadline (primary protocol P2, development cohort):")
print(f"  typed mean utility      {A3['T_mean_util']:+.4f}")
print(f"  monolithic mean utility {A3['M_mean_util']:+.4f}")
d = A3["paired_diff_T_minus_M"]
print(f"  typed - monolithic      {d[0]:+.4f} (95% CI {d[1]:+.4f} to {d[2]:+.4f})")
print("REPRODUCED" if ok else "NOT REPRODUCED")
sys.exit(0 if ok else 1)
