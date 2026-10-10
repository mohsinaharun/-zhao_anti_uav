# Waits for every agent checkpoint and for the running eval pass, then re-runs the (resumable)
# eval pass so A4 (all ops) and A5 (S1) are evaluated, then the analysis.
$root = Resolve-Path "$PSScriptRoot\..\.."
$res = "$root\restart\results\phase4"
$need = 'A4_P','A4_S2','A4_S1','A5_P','A5_S2','A5_S1' | ForEach-Object { "$res\agent_${_}_best.pt" }
while (($need | Where-Object { -not (Test-Path $_) }).Count -gt 0 -or
       -not (Select-String -Path "$res\eval.log" -Pattern 'EVAL PASS DONE' -Quiet)) { Start-Sleep 120 }
& pwsh -NoProfile -File "$PSScriptRoot\run_phase4_eval.ps1"
Set-Location $root
& "$HOME\Documents\anti_uav\Scripts\python.exe" "$PSScriptRoot\analyse_phase4.py" *>&1 | ForEach-Object { "$_" } | Set-Content "$res\analysis.txt"
"##### $(Get-Date -Format s) PHASE 4 FINISHED" | Add-Content "$res\eval.log"
