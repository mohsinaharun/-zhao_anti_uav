# Waits until the Phase 4g grid is complete (180 runs), then starts the three Phase 4h agents.
$root = Resolve-Path "$PSScriptRoot\..\.."
$g = "$root\restart\results\phase4g"
$log = "$root\restart\results\phase4h\launcher.log"
"##### $(Get-Date -Format s) waiting for Phase 4g" | Add-Content $log
while ((Get-ChildItem "$g\grid_*_seed*.json").Count -lt 180 -and -not (Test-Path "$g\STOPPED.json")) { Start-Sleep 60 }
if (Test-Path "$g\STOPPED.json") { "##### $(Get-Date -Format s) Phase 4g STOPPED; agents not started" | Add-Content $log; exit 1 }
Start-Sleep 120   # let the 4g workers exit
foreach ($op in 'P', 'S1', 'S2') {
    Start-Process pwsh -WindowStyle Hidden -ArgumentList '-NoProfile', '-File', "$PSScriptRoot\run_phase4h.ps1", '-Op', $op
}
"##### $(Get-Date -Format s) Phase 4g complete; started agents P, S1, S2" | Add-Content $log
