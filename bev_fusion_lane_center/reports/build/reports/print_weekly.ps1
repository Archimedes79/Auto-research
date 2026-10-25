# Print a weekly report HTML to PDF with headless Chrome.  Usage: powershell -File print_weekly.ps1 <html> <pdf>
param([Parameter(Mandatory=$true)][string]$Html, [Parameter(Mandatory=$true)][string]$Pdf)
$chrome = "C:\Program Files\Google\Chrome\Application\chrome.exe"
$profile = Join-Path $env:TEMP "chrome_pdf_profile"
if (Test-Path $Pdf) { Remove-Item $Pdf -Force }
$uri = "file:///" + ($Html -replace "\\", "/")
& $chrome --headless=new --disable-gpu --no-pdf-header-footer "--user-data-dir=$profile" "--print-to-pdf=$Pdf" $uri 2>$null
$t = 0
while (-not (Test-Path $Pdf) -and $t -lt 180) { Start-Sleep -Seconds 2; $t += 2 }
Start-Sleep -Seconds 2
if (Test-Path $Pdf) { "PDF written: $Pdf ({0:N1} MB)" -f ((Get-Item $Pdf).Length / 1MB) } else { "PDF NOT written" }
