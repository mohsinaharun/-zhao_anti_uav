# One-off: A4 at P in parallel (its slot in run_phase4_agents.ps1 crashed on the gpu_guard race).
$root = Resolve-Path "$PSScriptRoot\..\.."
$py = "$HOME\Documents\anti_uav\Scripts\python.exe"
$log = "$root\restart\results\phase4\train_A4P.log"
$env:PYTHONUNBUFFERED = 1; $env:GPU_SOFT_CAP = 540; Set-Location $root
"##### $(Get-Date -Format s) start A4 P" | Add-Content $log
& $py "$PSScriptRoot\phase4_agents.py" --arm A4 --op P *>&1 | ForEach-Object { "$_" } | Where-Object { $_ -match ' ep |done|Error|Traceback' } | Add-Content $log
"##### $(Get-Date -Format s) A4 P DONE" | Add-Content $log
