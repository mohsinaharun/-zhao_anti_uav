# Phase 4e, one operating point (search -> selection -> test eval). Resumable.
param([Parameter(Mandatory)][string]$Op)
$root = Resolve-Path "$PSScriptRoot\..\.."
$py = "$HOME\Documents\anti_uav\Scripts\python.exe"
$log = "$root\restart\results\phase4e\run_$Op.log"
$env:PYTHONUNBUFFERED = 1; $env:GPU_SOFT_CAP = 540; Set-Location $root
"##### $(Get-Date -Format s) start $Op" | Add-Content $log
& $py "$PSScriptRoot\phase4e.py" run --op $Op *>&1 | ForEach-Object { "$_" } | Where-Object { $_ -notmatch '^=+$|WARNING|Run in an ADMIN|A reboot|Falling back|This machine' } | Add-Content $log
"##### $(Get-Date -Format s) exit $Op" | Add-Content $log
