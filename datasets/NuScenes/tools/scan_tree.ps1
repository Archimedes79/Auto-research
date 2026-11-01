# scan_tree.ps1 - helper of make_manifest.py: the fingerprint of folders on D:, read natively (about 170,000 files per second; from WSL the same listing takes 45 s for 34,000 files).
# For every folder (relative to -Root, comma separated, "/" as separator): the number of files below it, their total bytes and the SHA-256 of the sorted list
# "<path relative to the folder, / as separator><TAB><size in bytes><LF>" (ordinal order of the paths, UTF-8 without BOM). make_manifest.py computes the same value in Python
# when this script cannot be used; the selftest of make_manifest.py checks both against a known answer.
# Output: one JSON array [{"folder", "files", "bytes", "list_sha256"}], a missing folder gives null values. Read-only.
# Usage: powershell.exe -NoProfile -ExecutionPolicy Bypass -File scan_tree.ps1 -Root D:\Auto-research\NuScenes -Folders nuscenes_full/samples/CAM_FRONT,nuscenes_full/maps
param(
    [Parameter(Mandatory = $true)][string]$Root,
    [Parameter(Mandatory = $true)][string]$Folders
)
$ErrorActionPreference = 'Stop'
$sha = [System.Security.Cryptography.SHA256]::Create()
$enc = New-Object System.Text.UTF8Encoding($false)
$res = @()
foreach ($f in ($Folders -split ',')) {
    $path = Join-Path $Root ($f -replace '/', '\')
    $dir = New-Object System.IO.DirectoryInfo $path
    if (-not $dir.Exists) {
        $res += [pscustomobject]@{ folder = $f; files = $null; bytes = $null; list_sha256 = $null }
        continue
    }
    $prefix = $dir.FullName.TrimEnd('\') + '\'
    $names = New-Object 'System.Collections.Generic.List[string]'
    $size = New-Object 'System.Collections.Generic.Dictionary[string,long]'
    $total = [int64]0
    foreach ($fi in $dir.EnumerateFiles('*', [System.IO.SearchOption]::AllDirectories)) {
        $rel = $fi.FullName.Substring($prefix.Length).Replace('\', '/')
        $names.Add($rel)
        $size[$rel] = $fi.Length
        $total += $fi.Length
    }
    $arr = $names.ToArray()
    [Array]::Sort($arr, [System.StringComparer]::Ordinal)
    $sb = New-Object System.Text.StringBuilder
    foreach ($n in $arr) { [void]$sb.Append($n).Append("`t").Append($size[$n]).Append("`n") }
    $hash = -join ($sha.ComputeHash($enc.GetBytes($sb.ToString())) | ForEach-Object { $_.ToString('x2') })
    $res += [pscustomobject]@{ folder = $f; files = $arr.Length; bytes = $total; list_sha256 = $hash }
}
ConvertTo-Json -InputObject @($res) -Compress
