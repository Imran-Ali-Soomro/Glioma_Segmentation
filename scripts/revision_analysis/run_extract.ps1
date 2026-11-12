param([string]$Set)
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$py = if ($env:PYTHON) { $env:PYTHON } else { "python" }
$d = $PSScriptRoot
New-Item -ItemType Directory -Force "$d\logs" | Out-Null
foreach ($f in 0..4) {
    & $py "$d\extract_regions_and_masks.py" --set $Set --fold $f 2>&1 | Out-File -Encoding utf8 "$d\logs\extract_${Set}_fold$f.log"
}
"ALL_DONE" | Out-File "$d\logs\extract_${Set}_DONE.txt"
