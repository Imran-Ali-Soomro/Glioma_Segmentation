param([string]$Set)
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$py = if ($env:PYTHON) { $env:PYTHON } else { "python" }
$d = $PSScriptRoot
foreach ($f in 0..4) {
    & $py "$d\extract_regions_and_masks.py" --set $Set --fold $f --infer sliding 2>&1 | Out-File -Encoding utf8 "$d\logs\extract_slide_${Set}_fold$f.log"
}
"ALL_DONE" | Out-File "$d\logs\extract_slide_${Set}_DONE.txt"
