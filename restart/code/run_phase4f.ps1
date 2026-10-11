# Phase 4f (VisDrone car), every stage (resumable). Usage: run_phase4f.ps1 [-Mode auto|prepare|analyse]
param([string]$Mode = 'auto')
$root = Resolve-Path "$PSScriptRoot\..\.."
$py = "$HOME\Documents\anti_uav\Scripts\python.exe"
$log = "$root\restart\results\phase4f\visdrone.log"
New-Item -ItemType Directory -Force "$root\restart\results\phase4f" | Out-Null
$env:PYTHONUNBUFFERED = 1; $env:GPU_SOFT_CAP = 540; Set-Location $root
"##### $(Get-Date -Format s) start $Mode" | Add-Content $log
& $py "$PSScriptRoot\phase4f.py" $Mode *>&1 | ForEach-Object { "$_" } | Where-Object { $_ -notmatch '^=+$|WARNING|Run in an ADMIN|A reboot|Falling back|This machine' } | Add-Content $log
"##### $(Get-Date -Format s) exit $Mode" | Add-Content $log
