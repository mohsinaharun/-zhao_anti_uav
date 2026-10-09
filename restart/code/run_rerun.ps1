# Re-runs with calibrated C*, as pre-registered in ..\rerun\PREREG.md. Resumable.
$ErrorActionPreference = 'Continue'
$root = Resolve-Path "$PSScriptRoot\..\.."
$py = "$HOME\Documents\anti_uav\Scripts\python.exe"; $r = "$PSScriptRoot\phase2_fl.py"
$log = "$root\restart\results\rerun\run.log"
New-Item -ItemType Directory -Force "$root\restart\results\rerun" | Out-Null
$env:PYTHONUNBUFFERED = 1; $env:GPU_SOFT_CAP = 540; Set-Location $root
$PID | Set-Content "$root\restart\results\rerun\run.pid"
function Run([string[]]$a) { & $py $r @a 2>&1 | Select-String -Pattern 'final|Error|Traceback' | ForEach-Object { $_.Line } | Add-Content $log }
foreach ($seed in 0..9) {
  # R1: central sigma sweep, B=64, C*
  foreach ($s in '0','0.5','1','1.5','2','2.5','3','3.5','4') {
    Run @('--mode','central','--dp','--sigma',$s,'--B','64','--clip','calibrated','--lr','0.03','--seed',"$seed",'--outdir','rerun') }
  # R2: true no-DP cells
  Run @('--mode','fl','--B','64','--local-iters','20','--mu','0','--lr','0.03','--seed',"$seed",'--outdir','rerun')
  Run @('--mode','central','--B','64','--lr','0.03','--seed',"$seed",'--outdir','rerun')
  # R3: FedProx mu sweep at sigma 1, 2
  foreach ($s in '1','2') { foreach ($mu in '0.01','0.03','0.1','0.3','1') {
    Run @('--mode','fl','--dp','--sigma',$s,'--B','64','--local-iters','20','--mu',$mu,'--clip','calibrated','--lr','0.03','--seed',"$seed",'--outdir','rerun') } }
  "##### $(Get-Date -Format s) seed $seed done" | Add-Content $log
}
"##### RERUN COMPLETE $(Get-Date -Format s)" | Add-Content $log
