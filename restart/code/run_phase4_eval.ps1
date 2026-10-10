# Phase 4 TEST evaluation (PREREG §6): seeds 200-219. Fixed arms always; A4/A5 only where the
# deployed checkpoint exists. Resumable (finished seeds are skipped) - re-run after training ends.
$root = Resolve-Path "$PSScriptRoot\..\.."
$py = "$HOME\Documents\anti_uav\Scripts\python.exe"
$res = "$root\restart\results\phase4"
$log = "$res\eval.log"
$env:PYTHONUNBUFFERED = 1; $env:GPU_SOFT_CAP = 540; Set-Location $root
foreach ($op in 'P','S2','S1') {
  foreach ($arm in 'A1','A2','A3','A4','A5') {
    $extra = @()
    if ($arm -in 'A4','A5') {
      $pt = "$res\agent_${arm}_${op}_best.pt"
      if (-not (Test-Path $pt)) { continue }
      $extra = @('--policy', $pt)
    }
    "##### $(Get-Date -Format s) eval $op $arm" | Add-Content $log
    & $py "$PSScriptRoot\phase4.py" eval --op $op --arm $arm --seeds 200-219 @extra *>&1 | ForEach-Object { "$_" } | Where-Object { $_ -match 'eval_|Error|Traceback' } | Add-Content $log
  }
}
"##### $(Get-Date -Format s) EVAL PASS DONE" | Add-Content $log
