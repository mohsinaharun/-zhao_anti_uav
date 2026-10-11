# Phase 4h, one operating point (gate [P only], train A5-relative, evaluate). Usage: run_phase4h.ps1 -Op P
param([string]$Op = 'P')
$root = Resolve-Path "$PSScriptRoot\..\.."
$py = "$HOME\Documents\anti_uav\Scripts\python.exe"
$log = "$root\restart\results\phase4h\agent_$Op.log"
New-Item -ItemType Directory -Force "$root\restart\results\phase4h" | Out-Null
$env:PYTHONUNBUFFERED = 1; $env:GPU_SOFT_CAP = 540; Set-Location $root
"##### $(Get-Date -Format s) start op $Op" | Add-Content $log
& $py "$PSScriptRoot\phase4h.py" go --op $Op *>&1 | ForEach-Object { "$_" } | Where-Object { $_ -notmatch '^=+$|WARNING|Run in an ADMIN|A reboot|Falling back|This machine' } | Add-Content $log
"##### $(Get-Date -Format s) exit op $Op" | Add-Content $log
