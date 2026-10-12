# Phase 5d grid, one worker (resumable). Usage: run_phase5d.ps1 -Worker 0 -Workers 3
param([int]$Worker = 0, [int]$Workers = 1)
$root = Resolve-Path "$PSScriptRoot\..\.."
$py = "$HOME\Documents\anti_uav\Scripts\python.exe"
$log = "$root\restart\results\phase5d\worker$Worker.log"
New-Item -ItemType Directory -Force "$root\restart\results\phase5d" | Out-Null
$env:PYTHONUNBUFFERED = 1; $env:GPU_SOFT_CAP = 540; Set-Location $root
"##### $(Get-Date -Format s) start worker $Worker/$Workers" | Add-Content $log
& $py "$PSScriptRoot\phase5d.py" run --worker $Worker --workers $Workers *>&1 | ForEach-Object { "$_" } | Where-Object { $_ -notmatch '^=+$|WARNING|Run in an ADMIN|A reboot|Falling back|This machine' } | Add-Content $log
"##### $(Get-Date -Format s) exit worker $Worker" | Add-Content $log
