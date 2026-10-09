# Phase 4 agent training for ONE arm (A4 or A5), operating points P -> S2 -> S1.
# Usage: run_phase4_agents.ps1 -Arm A5   (launch detached; each op resumable only as a whole)
param([Parameter(Mandatory)][string]$Arm)
$root = Resolve-Path "$PSScriptRoot\..\.."
$py = "$HOME\Documents\anti_uav\Scripts\python.exe"
$log = "$root\restart\results\phase4\train_$Arm.log"
New-Item -ItemType Directory -Force "$root\restart\results\phase4" | Out-Null
$env:PYTHONUNBUFFERED = 1; $env:GPU_SOFT_CAP = 540; Set-Location $root
foreach ($op in 'P','S2','S1') {
  if (Test-Path "$root\restart\results\phase4\agent_${Arm}_${op}_best.pt") { continue }
  "##### $(Get-Date -Format s) start $Arm $op" | Add-Content $log
  & $py "$PSScriptRoot\phase4_agents.py" --arm $Arm --op $op 2>&1 | Select-String -Pattern ' ep |done|Error|Traceback' | ForEach-Object { $_.Line } | Add-Content $log
}
"##### $(Get-Date -Format s) $Arm ALL DONE" | Add-Content $log
