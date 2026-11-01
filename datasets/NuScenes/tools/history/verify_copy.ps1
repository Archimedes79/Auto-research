# One-off (2026-10-25): check a copy against its manifest (tab-separated: relative path, size in bytes).
# Every manifest file must exist under the target with the same size; prints counts and the first mismatches.
# Usage: powershell -File verify_copy.ps1 <manifest.tsv> <target dir>
param([string]$Manifest, [string]$Target)
$want = @{}
foreach ($line in [IO.File]::ReadLines($Manifest)) {
    $parts = $line.Split("`t")
    $rel = $parts[0]; if ($rel.StartsWith('./')) { $rel = $rel.Substring(2) }
    $want[$rel.Replace('/', '\')] = [int64]$parts[1]
}
$root = (Resolve-Path $Target).Path.TrimEnd('\') + '\'
$have = @{}
Get-ChildItem -LiteralPath $root -Recurse -File -Force | ForEach-Object { $have[$_.FullName.Substring($root.Length)] = $_.Length }
$missing = @(); $wrong = @()
foreach ($k in $want.Keys) {
    if (-not $have.ContainsKey($k)) { $missing += $k } elseif ($have[$k] -ne $want[$k]) { $wrong += $k }
}
"manifest files: $($want.Count); target files: $($have.Count); missing: $($missing.Count); size mismatch: $($wrong.Count)"
$missing | Select-Object -First 5 | ForEach-Object { "missing: $_" }
$wrong | Select-Object -First 5 | ForEach-Object { "size: $_" }
if ($missing.Count -eq 0 -and $wrong.Count -eq 0) { "VERIFY OK" } else { "VERIFY FAILED"; exit 1 }
