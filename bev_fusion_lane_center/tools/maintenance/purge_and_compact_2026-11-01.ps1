# Purge and compact, prepared 2026-11-01 (lane-centre system). Run by Mirko in an ELEVATED PowerShell; nothing in it has run:
#   powershell -ExecutionPolicy Bypass -File .\bev_fusion_lane_center\tools\maintenance\purge_and_compact_2026-11-01.ps1
# Phase 1 deletes (asks first): the lane archive on D:, the WSL caches of D5 (fused32, mini copies, review rounds) and the second-wave list (69 files).
# Phase 2 compacts the WSL image (asks first): wsl --shutdown KILLS every WSL job (other chats' trainings and tests too), so it lists them first.
# -NoCompact stops after phase 1 (no administrator rights needed). pert / p17 (74.6 GB) are NOT touched: that is Mirko's decision.
param([switch]$NoCompact, [string]$Archive = 'D:\Auto-research\lane_archive_2026-11-01')
$ErrorActionPreference = 'Stop'
$distro = 'Ubuntu-22.04'
$here = $PSScriptRoot
$wslHere = (wsl -d $distro wslpath -a "$here").Trim()
$base = (Get-ItemProperty 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Lxss\*' | Where-Object { $_.DistributionName -eq $distro }).BasePath
$vhd = Join-Path $base 'ext4.vhdx'
function Report($t) { '{0}: C: free {1:N1} GB, ext4.vhdx {2:N1} GB' -f $t, ((Get-PSDrive C).Free / 1GB), ((Get-Item $vhd).Length / 1GB) }
$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $NoCompact -and -not $isAdmin) { throw 'Compaction needs an elevated PowerShell (Run as administrator), or use -NoCompact for the deletes only.' }
Report 'before'

# what a shutdown would kill
$jobs = wsl -d $distro -e bash -c "ps -eo pid,etime,args | grep -E 'train.py|evaluate.py|run.py|selftest|autogt|test_' | grep -v grep | cut -c1-150"
if ($jobs) { 'Running in WSL now (a shutdown kills these):'; $jobs }

# phase 1: deletes
'--- dry run of the WSL lists'
wsl -d $distro bash "$wslHere/wsl_purge_d5_2026-11-01.sh" | Select-Object -Last 1
wsl -d $distro bash "$wslHere/wsl_second_wave_2026-11-01.sh" | Select-Object -Last 1
if (Test-Path $Archive) { 'Archive on D: ({0:N1} GB): {1}' -f ((Get-ChildItem $Archive -Recurse -File | Measure-Object Length -Sum).Sum / 1GB), $Archive }
if ((Read-Host 'Delete the lane archive on D: and the WSL caches listed above? (type yes)') -eq 'yes') {
    if ((Test-Path $Archive) -and ($Archive -like '*lane_archive_2026-11-01*')) { Remove-Item -Recurse -Force $Archive; 'archive deleted' }
    wsl -d $distro bash "$wslHere/wsl_purge_d5_2026-11-01.sh" --delete | Select-Object -Last 1
    wsl -d $distro bash "$wslHere/wsl_second_wave_2026-11-01.sh" --delete | Select-Object -Last 1
} else { 'phase 1 skipped' }

# phase 2: compaction
if ($NoCompact) { Report 'after (no compaction)'; return }
if ((Read-Host 'Shut WSL down now and compact the image? Every running WSL job ends. (type yes)') -ne 'yes') { Report 'after (compaction skipped)'; return }
wsl -d $distro -u root -e fstrim -av 2>$null | Out-Null   # tell the virtual disk which blocks are free (harmless if unsupported)
wsl --shutdown
Start-Sleep -Seconds 8
if (Get-Command Optimize-VHD -ErrorAction SilentlyContinue) {
    Optimize-VHD -Path $vhd -Mode Full
} else {
    $dp = Join-Path $env:TEMP 'compact_ext4.txt'
    "select vdisk file=`"$vhd`"`r`nattach vdisk readonly`r`ncompact vdisk`r`ndetach vdisk`r`nexit" | Set-Content $dp -Encoding ascii
    diskpart /s $dp
}
wsl -d $distro -e true   # start the distro again
Report 'after'
