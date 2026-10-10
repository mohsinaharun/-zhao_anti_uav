# Phase 4c, one dataset, every stage (resumable). Usage: run_phase4c.ps1 -Task cifar10|voc
param([Parameter(Mandatory)][string]$Task)
$root = Resolve-Path "$PSScriptRoot\..\.."
$py = "$HOME\Documents\anti_uav\Scripts\python.exe"
$log = "$root\restart\results\phase4c\$Task.log"
New-Item -ItemType Directory -Force "$root\restart\results\phase4c" | Out-Null
$env:PYTHONUNBUFFERED = 1; $env:GPU_SOFT_CAP = 540; Set-Location $root
"##### $(Get-Date -Format s) start $Task" | Add-Content $log
& $py "$PSScriptRoot\phase4c.py" auto --task $Task *>&1 | ForEach-Object { "$_" } | Where-Object { $_ -notmatch '^=+$|WARNING|Run in an ADMIN|A reboot|Falling back|This machine' } | Add-Content $log
"##### $(Get-Date -Format s) exit $Task" | Add-Content $log
