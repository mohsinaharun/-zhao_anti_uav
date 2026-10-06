# Phase 3 extended noise-pulse test, as pre-registered in ..\phase3\PREREG.md.
# V1 -> V2 -> V3. Resumable: every run is skipped if its JSON exists. Launch detached:
#   Start-Process pwsh -ArgumentList '-NoProfile','-File',"<this file>" -WindowStyle Hidden
$ErrorActionPreference = 'Continue'
$root = Resolve-Path "$PSScriptRoot\..\.."
$py   = "$HOME\Documents\anti_uav\Scripts\python.exe"
$r    = "$PSScriptRoot\phase2_fl.py"
$log  = "$root\restart\results\phase3\run.log"
New-Item -ItemType Directory -Force "$root\restart\results\phase3" | Out-Null
$env:PYTHONUNBUFFERED = 1; $env:GPU_SOFT_CAP = 540
Set-Location $root
$PID | Set-Content "$root\restart\results\phase3\run.pid"
$common = @('--mode','fl','--dp','--B','64','--local-iters','20','--mu','0','--lr','0.03','--clip','calibrated','--outdir','phase3')
foreach ($v in @(@{n='V1';R=12;K=9}, @{n='V2';R=25;K=9}, @{n='V3';R=12;K=7})) {
  "##### $(Get-Date -Format s) start $($v.n)" | Add-Content $log
  foreach ($seed in 0..19) {
    foreach ($s in '1','2','4') {
      foreach ($k in 0..$v.R) {
        $a = $common + @('--sigma',$s,'--rounds',"$($v.R)",'--K',"$($v.K)",'--seed',"$seed")
        if ($k -gt 0) { $a += @('--pulse-round',"$k") }
        & $py $r @a 2>&1 | Select-String -Pattern 'final|Error|Traceback' | ForEach-Object { $_.Line } | Add-Content $log
      }
    }
    "##### $(Get-Date -Format s) $($v.n) seed $seed done" | Add-Content $log
  }
  "##### $(Get-Date -Format s) $($v.n) COMPLETE" | Add-Content $log
}
"##### PHASE 3 COMPLETE $(Get-Date -Format s)" | Add-Content $log
