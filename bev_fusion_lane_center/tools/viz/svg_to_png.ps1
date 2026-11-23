# svg_to_png: a PNG of an SVG drawing (for decks and mails that cannot show SVG) through a headless Microsoft Edge screenshot; read-only on the SVG, one short process.
# Promoted 2026-11-23 from the report scratch of 2026-11-03 (D140). Windows only (Edge is part of Windows 11).
#   powershell -File tools\viz\svg_to_png.ps1 -svg <in.svg> -png <out.png> [-w 1600] [-h 900]
#   powershell -File tools\viz\svg_to_png.ps1 -selftest           a tiny SVG to a PNG in %TEMP%, checks the PNG signature
param([string]$svg, [string]$png, [int]$w = 1600, [int]$h = 900, [switch]$selftest)
$edge = @("${env:ProgramFiles(x86)}\Microsoft\Edge\Application\msedge.exe", "$env:ProgramFiles\Microsoft\Edge\Application\msedge.exe") | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $edge) { Write-Output "svg_to_png: Microsoft Edge not found"; exit 2 }

function Convert-Svg([string]$in, [string]$out, [int]$width, [int]$height) {
    $full = (Resolve-Path $in).Path
    $uri = "file:///" + ($full -replace '\\', '/')
    $prof = Join-Path $env:TEMP "svg_to_png_edge_profile"
    $outFull = [System.IO.Path]::GetFullPath($out)
    & $edge --headless=new --disable-gpu --no-first-run --user-data-dir="$prof" --hide-scrollbars "--window-size=$width,$height" "--screenshot=$outFull" $uri | Out-Null
    for ($i = 0; $i -lt 20 -and -not (Test-Path $outFull); $i++) { Start-Sleep -Milliseconds 250 }
    return (Test-Path $outFull)
}

if ($selftest) {
    $d = Join-Path $env:TEMP ("svg_to_png_selftest_" + [guid]::NewGuid().ToString("N").Substring(0, 8))
    New-Item -ItemType Directory -Path $d | Out-Null
    $s = Join-Path $d "t.svg"; $p = Join-Path $d "t.png"
    Set-Content -Path $s -Encoding utf8 -Value '<svg xmlns="http://www.w3.org/2000/svg" width="200" height="100"><rect width="200" height="100" fill="#1a9a2a"/></svg>'
    $ok = Convert-Svg $s $p 200 100
    $sig = if ($ok) { [System.IO.File]::ReadAllBytes($p)[0..3] -join "," } else { "" }
    Remove-Item -Recurse -Force -Confirm:$false $d
    if ($ok -and $sig -eq "137,80,78,71") { Write-Output "svg_to_png selftest ok: a PNG was written (signature $sig)"; exit 0 }
    Write-Output "svg_to_png selftest FAIL (written: $ok, signature: $sig)"; exit 1
}
if (-not $svg -or -not $png) { Write-Output "usage: svg_to_png.ps1 -svg <in.svg> -png <out.png> [-w 1600] [-h 900] | -selftest"; exit 2 }
if (Convert-Svg $svg $png $w $h) { Write-Output "written $png" } else { Write-Output "svg_to_png: no PNG written"; exit 1 }
