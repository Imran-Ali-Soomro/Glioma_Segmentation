$env:KMP_DUPLICATE_LIB_OK = "TRUE"; $env:REV_VARIANT = "slide"
$py = if ($env:PYTHON) { $env:PYTHON } else { "python" }
$d = $PSScriptRoot
$jobs = @("P2:unw","P2:size","P1:unw","P1:size","P0:unw")
$procs = foreach ($j in $jobs) {
    $n = $j.Replace(":","_")
    Start-Process -FilePath $py -ArgumentList "$d\a2_run_protocols.py",$j -RedirectStandardOutput "$d\logs\a2slide_$n.log" -RedirectStandardError "$d\logs\a2slide_$n.err" -PassThru -WindowStyle Hidden
}
$procs | Wait-Process
"ALL_DONE" | Out-File "$d\logs\a2slide_DONE.txt"
