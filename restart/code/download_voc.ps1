# Resumable download of Pascal VOC 2007 trainval/test + 2012 trainval (Phase 4c). Retries until complete.
$dst = Resolve-Path "$PSScriptRoot\..\cache\ext\voc"
$files = @{
  'VOCtrainval_06-Nov-2007.tar' = 'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar';
  'VOCtest_06-Nov-2007.tar'     = 'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar';
  'VOCtrainval_11-May-2012.tar' = 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar' }
$size = @{ 'VOCtrainval_06-Nov-2007.tar' = 460032000; 'VOCtest_06-Nov-2007.tar' = 451020800; 'VOCtrainval_11-May-2012.tar' = 1999639040 }
foreach ($f in $files.Keys) {
  $p = Join-Path $dst $f
  for ($i = 0; $i -lt 200; $i++) {
    if ((Test-Path $p) -and (Get-Item $p).Length -eq $size[$f]) { break }
    & curl.exe -sS -L -C - --retry 5 -o $p $files[$f]
    Start-Sleep 5
  }
  "$(Get-Date -Format s) $f $((Get-Item $p).Length)" | Add-Content (Join-Path $dst 'download.log')
}
"$(Get-Date -Format s) ALL DONE" | Add-Content (Join-Path $dst 'download.log')
