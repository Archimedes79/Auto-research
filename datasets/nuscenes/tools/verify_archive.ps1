<#
.SYNOPSIS
  Compare nuScenes archives (.tgz / .tar.gz) member by member with the folder they were unpacked into. Native (Windows), about 50 times faster than
  `tar --diff` through WSL: it reads the archive once (gunzip, tar headers) and every member's file next to it.

.DESCRIPTION
  For every regular file in the archive: the file must exist below -Dest, have the same size, and the same bytes. An archive may be deleted only when this
  shows 0 differences (README, "Archives"); the manifest keeps the result (make_manifest.py --archive-check). Members outside -Only are skipped and counted
  (skipped_members / skipped_bytes: what the archive holds that is not on disk, by choice). Unsafe member names (absolute, "..") are differences.
  Understands ustar, GNU long names (L) and pax headers (x, g); a header whose checksum is wrong, a truncated or corrupt archive are differences too.

  verify_archive.ps1 -Archive <archive> -Dest <folder> [-Only samples/CAM_FRONT/,sweeps/CAM_FRONT/] [-OutDir <folder>]
  verify_archive.ps1 -List <file> [-Parallel 4] [-OutDir <folder>]     one line per archive: archive|dest|prefix1;prefix2 (prefixes optional, with a trailing /)
  verify_archive.ps1 -SelfTest                                         archives made by bsdtar and GNU tar (gnu / pax / ustar), long names, tampering

  Prints one JSON line per archive: {"archive", "members", "files", "bytes", "differences", "missing", "size_diff", "content_diff", "unsafe", "bad_header",
  "truncated", "skipped_members", "skipped_bytes", "seconds", "date"}; with -OutDir also <name>.json and <name>.problems.txt (the first 200 differences).
  Exit code 0 when every archive has 0 differences and nothing was skipped; 1 when an archive has a difference; 2 on a wrong call; 3 when there is no difference but
  members were skipped (-Only): a partial comparison, which does not allow deleting the archive (the skipped members are not on disk).
  -Only takes prefixes separated by comma or semicolon (from bash / cmd a comma list arrives as ONE string and is split here); -List ignores -Only (its lines carry their own).
#>
param([string]$Archive, [string]$Dest, [string[]]$Only, [string]$List, [int]$Parallel = 4, [string]$OutDir, [switch]$SelfTest)
$ErrorActionPreference = 'Stop'
if ($Only) { $Only = @($Only | ForEach-Object { $_ -split '[,;]' } | Where-Object { $_ }) }      # from bash a comma list arrives as ONE string
try { (Get-Process -Id $PID).PriorityClass = 'BelowNormal' } catch { }

$src = @'
using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

// the archive file as it is read: every byte that goes to the gunzip is hashed too (the identity of the exact file that was compared)
public class HashStream : Stream
{
    Stream inner;
    System.Security.Cryptography.SHA256 sha = System.Security.Cryptography.SHA256.Create();
    public long Count;
    public HashStream(Stream s) { inner = s; }
    public override int Read(byte[] buffer, int offset, int count)
    {
        int n = inner.Read(buffer, offset, count);
        if (n > 0) { sha.TransformBlock(buffer, offset, n, null, 0); Count += n; }
        return n;
    }
    public string Finish()                                             // read (and hash) what is left of the file, return the hex digest
    {
        byte[] scratch = new byte[1 << 20];
        int n;
        while ((n = inner.Read(scratch, 0, scratch.Length)) > 0) { sha.TransformBlock(scratch, 0, n, null, 0); Count += n; }
        sha.TransformFinalBlock(new byte[0], 0, 0);
        return BitConverter.ToString(sha.Hash).Replace("-", "").ToLowerInvariant();
    }
    public override bool CanRead { get { return true; } }
    public override bool CanSeek { get { return false; } }
    public override bool CanWrite { get { return false; } }
    public override long Length { get { throw new NotSupportedException(); } }
    public override long Position { get { throw new NotSupportedException(); } set { throw new NotSupportedException(); } }
    public override void Flush() { }
    public override long Seek(long offset, SeekOrigin origin) { throw new NotSupportedException(); }
    public override void SetLength(long value) { throw new NotSupportedException(); }
    public override void Write(byte[] buffer, int offset, int count) { throw new NotSupportedException(); }
}

public static class Crc32
{
    public static readonly uint[] Table = MakeTable();
    static uint[] MakeTable()
    {
        uint[] t = new uint[256];
        for (uint n = 0; n < 256; n++)
        {
            uint c = n;
            for (int k = 0; k < 8; k++) c = ((c & 1) != 0) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
            t[n] = c;
        }
        return t;
    }
}

// the decoded tar stream: counts what is read from it and keeps its CRC-32 (compared with the gzip trailer at the end)
public class CountStream : Stream
{
    Stream inner;
    public long Count;
    uint crc = 0xFFFFFFFFu;
    public uint Crc { get { return ~crc; } }
    public CountStream(Stream s) { inner = s; }
    public override int Read(byte[] buffer, int offset, int count)
    {
        int n = inner.Read(buffer, offset, count);
        if (n > 0)
        {
            Count += n;
            uint c = crc;
            uint[] tab = Crc32.Table;
            for (int i = offset; i < offset + n; i++) c = tab[(c ^ buffer[i]) & 0xFF] ^ (c >> 8);
            crc = c;
        }
        return n;
    }
    public override bool CanRead { get { return true; } }
    public override bool CanSeek { get { return false; } }
    public override bool CanWrite { get { return false; } }
    public override long Length { get { throw new NotSupportedException(); } }
    public override long Position { get { throw new NotSupportedException(); } set { throw new NotSupportedException(); } }
    public override void Flush() { }
    public override long Seek(long offset, SeekOrigin origin) { throw new NotSupportedException(); }
    public override void SetLength(long value) { throw new NotSupportedException(); }
    public override void Write(byte[] buffer, int offset, int count) { throw new NotSupportedException(); }
}

public class ArchiveResult
{
    public string Archive, Sha256;
    public long Members, Files, Bytes, Missing, SizeDiff, ContentDiff, Unsafe, BadHeader, Other, SkippedMembers, SkippedBytes, ArchiveBytes;
    public bool Truncated, NothingCompared;
    public double Seconds;
    public List<string> Problems = new List<string>();
    public long Differences { get { return Missing + SizeDiff + ContentDiff + Unsafe + BadHeader + (Truncated ? 1 : 0) + (NothingCompared ? 1 : 0) + Other; } }
    public void Add(string kind, string name) { if (Problems.Count < 200) Problems.Add(kind + ": " + name); }
    public string Json()
    {
        return "{\"archive\": \"" + Archive.Replace("\\", "/") + "\", \"members\": " + Members + ", \"files\": " + Files + ", \"bytes\": " + Bytes +
               ", \"differences\": " + Differences + ", \"missing\": " + Missing + ", \"size_diff\": " + SizeDiff + ", \"content_diff\": " + ContentDiff +
               ", \"unsafe\": " + Unsafe + ", \"other\": " + Other + ", \"bad_header\": " + BadHeader + ", \"truncated\": " + (Truncated ? "true" : "false") + ", \"nothing_compared\": " + (NothingCompared ? "true" : "false") +
               ", \"skipped_members\": " + SkippedMembers + ", \"skipped_bytes\": " + SkippedBytes +
               ", \"archive_bytes\": " + ArchiveBytes + ", \"archive_sha256\": " + (Sha256 == null ? "null" : "\"" + Sha256 + "\"") +
               ", \"seconds\": " + Seconds.ToString("0.0", System.Globalization.CultureInfo.InvariantCulture) +
               ", \"date\": \"" + DateTime.Now.ToString("yyyy-MM-dd", System.Globalization.CultureInfo.InvariantCulture) + "\"}";
    }
}

public static class ArchiveCheck
{
    [DllImport("msvcrt.dll", CallingConvention = CallingConvention.Cdecl)]
    static extern int memcmp(byte[] a, byte[] b, UIntPtr count);

    static void GzipTrailer(string path, out uint crc, out uint isize)          // the last 8 bytes of a gzip file: CRC-32 and length (mod 2^32) of the LAST member
    {
        crc = 0; isize = 0;
        using (FileStream f = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite))
        {
            if (f.Length < 8) return;
            f.Seek(-8, SeekOrigin.End);
            byte[] b = new byte[8];
            ReadFull(f, b, 0, 8);
            crc = (uint)b[0] | ((uint)b[1] << 8) | ((uint)b[2] << 16) | ((uint)b[3] << 24);
            isize = (uint)b[4] | ((uint)b[5] << 8) | ((uint)b[6] << 16) | ((uint)b[7] << 24);
        }
    }

    static int ReadFull(Stream s, byte[] buf, int off, int n)
    {
        int got = 0;
        while (got < n)
        {
            int r = s.Read(buf, off + got, n - got);
            if (r <= 0) break;
            got += r;
        }
        return got;
    }

    static void Skip(Stream s, long n, byte[] scratch)
    {
        while (n > 0)
        {
            int r = s.Read(scratch, 0, (int)Math.Min(n, (long)scratch.Length));
            if (r <= 0) throw new EndOfStreamException("the archive ends inside a member");
            n -= r;
        }
    }

    static long ParseNum(byte[] b, int off, int len)
    {
        if ((b[off] & 0x80) != 0)                                  // base-256 (sizes of 8 GiB and more)
        {
            long v = b[off] & 0x7F;
            for (int i = 1; i < len; i++) v = (v << 8) | b[off + i];
            return v;
        }
        long r = 0;
        int k = 0;
        while (k < len && (b[off + k] == (byte)' ' || b[off + k] == 0)) k++;
        for (; k < len; k++)
        {
            byte c = b[off + k];
            if (c < (byte)'0' || c > (byte)'7') break;
            r = r * 8 + (c - (byte)'0');
        }
        return r;
    }

    static string CStr(byte[] b, int off, int len)
    {
        int n = 0;
        while (n < len && b[off + n] != 0) n++;
        return Encoding.UTF8.GetString(b, off, n);
    }

    static bool HeaderOk(byte[] h)
    {
        long want = ParseNum(h, 148, 8);
        long sum = 0, ssum = 0;
        for (int i = 0; i < 512; i++)
        {
            byte v = (i >= 148 && i < 156) ? (byte)' ' : h[i];
            sum += v;
            ssum += (sbyte)v;
        }
        return sum == want || ssum == want;
    }

    static void ParsePax(byte[] data, ref string path, ref long size)
    {
        int pos = 0;
        while (pos < data.Length)
        {
            int sp = pos;
            while (sp < data.Length && data[sp] != (byte)' ') sp++;
            int recLen;
            if (!int.TryParse(Encoding.ASCII.GetString(data, pos, sp - pos), out recLen) || recLen <= 0 || pos + recLen > data.Length) break;
            int end = pos + recLen;
            string rec = Encoding.UTF8.GetString(data, sp + 1, Math.Max(0, end - sp - 2));      // "<len> key=value\n"
            int eq = rec.IndexOf('=');
            if (eq > 0)
            {
                string k = rec.Substring(0, eq), v = rec.Substring(eq + 1);
                if (k == "path") path = v;
                else if (k == "size") { long s; if (long.TryParse(v, out s)) size = s; }
            }
            pos = end;
        }
    }

    static bool Unsafe(string name)
    {
        if (name.StartsWith("/") || name.StartsWith("\\") || (name.Length > 1 && name[1] == ':')) return true;
        foreach (string part in name.Split('/', '\\')) if (part == "..") return true;
        return false;
    }

    static bool Wanted(string name, string[] only)
    {
        if (only == null || only.Length == 0) return true;
        foreach (string p in only) if (name.StartsWith(p, StringComparison.Ordinal)) return true;
        return false;
    }

    public static ArchiveResult Check(string archive, string root, string[] only)
    {
        ArchiveResult res = new ArchiveResult();
        res.Archive = Path.GetFileName(archive);
        System.Diagnostics.Stopwatch sw = System.Diagnostics.Stopwatch.StartNew();
        byte[] hdr = new byte[512], a = new byte[1 << 20], b = new byte[1 << 20], scratch = new byte[1 << 16];
        string longName = null, paxPath = null;
        long paxSize = -1;
        try
        {
            using (FileStream fs = new FileStream(archive, FileMode.Open, FileAccess.Read, FileShare.Read, 1 << 20))
            using (HashStream hs = new HashStream(fs))
            using (GZipStream gzs = new GZipStream(hs, CompressionMode.Decompress))
            {
                CountStream gz = new CountStream(gzs);
                int zeroBlocks = 0;
                bool ended = false;
                while (true)
                {
                    int got = ReadFull(gz, hdr, 0, 512);
                    if (got == 0) break;
                    if (got < 512) { res.Truncated = true; res.Add("truncated", "header after member " + res.Members); break; }
                    bool zero = true;
                    for (int i = 0; i < 512; i++) if (hdr[i] != 0) { zero = false; break; }
                    if (zero) { zeroBlocks++; if (zeroBlocks >= 2) { ended = true; break; } continue; }
                    zeroBlocks = 0;
                    if (!HeaderOk(hdr)) { res.BadHeader++; res.Add("bad header checksum", "after member " + res.Members); break; }
                    char type = (char)hdr[156];
                    long size = ParseNum(hdr, 124, 12);
                    long pad = (512 - size % 512) % 512;
                    if (type == 'L' || type == 'K' || type == 'x' || type == 'g')
                    {
                        if (size > (1 << 24)) { res.BadHeader++; res.Add("extended header too large", "after member " + res.Members); break; }
                        byte[] data = new byte[size];
                        if (ReadFull(gz, data, 0, (int)size) < size) throw new EndOfStreamException("the archive ends inside an extended header");
                        Skip(gz, pad, scratch);
                        if (type == 'L') longName = CStr(data, 0, data.Length);
                        else if (type == 'x') ParsePax(data, ref paxPath, ref paxSize);
                        continue;
                    }
                    string name = CStr(hdr, 0, 100);
                    if (hdr[257] == (byte)'u' && hdr[258] == (byte)'s' && hdr[259] == (byte)'t' && hdr[260] == (byte)'a' && hdr[261] == (byte)'r')
                    {
                        string prefix = CStr(hdr, 345, 155);
                        if (prefix.Length > 0) name = prefix + "/" + name;
                    }
                    if (paxPath != null) name = paxPath; else if (longName != null) name = longName;
                    while (name.StartsWith("./")) name = name.Substring(2);                    // './samples/x' and 'samples/x' are the same member
                    if (paxSize >= 0) { size = paxSize; pad = (512 - size % 512) % 512; }
                    paxPath = null; longName = null; paxSize = -1;
                    res.Members++;
                    bool regular = type == '0' || type == '\0' || type == '7';
                    if (!regular)
                    {
                        if (type != '5') { res.Other++; res.Add("member type '" + type + "' (link, device, fifo, sparse ...) is not compared", name); }
                        Skip(gz, size + pad, scratch);
                        continue;
                    }
                    if (Unsafe(name)) { res.Unsafe++; res.Add("unsafe name", name); Skip(gz, size + pad, scratch); continue; }
                    if (!Wanted(name, only)) { res.SkippedMembers++; res.SkippedBytes += size; Skip(gz, size + pad, scratch); continue; }
                    string p = Path.Combine(root, name.Replace('/', Path.DirectorySeparatorChar));
                    FileInfo fi = new FileInfo(p);
                    if (!fi.Exists) { res.Missing++; res.Add("missing", name); Skip(gz, size + pad, scratch); continue; }
                    if (fi.Length != size) { res.SizeDiff++; res.Add("size " + fi.Length + " on disk, " + size + " in the archive", name); Skip(gz, size + pad, scratch); continue; }
                    bool same = true;
                    using (FileStream f = new FileStream(p, FileMode.Open, FileAccess.Read, FileShare.ReadWrite, 1 << 16))
                    {
                        long left = size;
                        while (left > 0)
                        {
                            int n = (int)Math.Min(left, (long)a.Length);
                            if (ReadFull(gz, a, 0, n) < n) throw new EndOfStreamException("the archive ends inside a member");
                            if (same && (ReadFull(f, b, 0, n) < n || memcmp(a, b, (UIntPtr)n) != 0)) same = false;
                            left -= n;
                        }
                    }
                    Skip(gz, pad, scratch);
                    res.Files++; res.Bytes += size;
                    if (!same) { res.ContentDiff++; res.Add("content differs", name); }
                }
                if (!ended && !res.Truncated && res.BadHeader == 0) { res.Truncated = true; res.Add("truncated", "the archive ends without an end-of-archive marker (two zero blocks) after member " + res.Members); }
                if (res.Files == 0 && !res.Truncated && res.BadHeader == 0) { res.NothingCompared = true; res.Add("nothing compared", res.Members + " members, " + res.SkippedMembers + " skipped, 0 regular files compared"); }
                bool nonzero = false;                                        // behind the end marker only zeros (padding to the record size) may follow
                int rd;
                while ((rd = gz.Read(scratch, 0, scratch.Length)) > 0) { if (!nonzero) for (int i = 0; i < rd; i++) if (scratch[i] != 0) { nonzero = true; break; } }
                if (nonzero) { res.Truncated = true; res.Add("data after the end-of-archive marker", "non-zero bytes behind the marker: members no tool would extract"); }
                res.Sha256 = hs.Finish();                                    // and the file to its last byte
                res.ArchiveBytes = hs.Count;
                uint tcrc, isize;                                            // the trailer of the LAST gzip member must be that of the one stream decoded here
                GzipTrailer(archive, out tcrc, out isize);
                if (isize != (uint)(gz.Count & 0xFFFFFFFF) || tcrc != gz.Crc) { res.Truncated = true; res.Add("gzip trailer", "decoded " + gz.Count + " bytes with CRC-32 " + gz.Crc.ToString("x8") + ", the last 8 bytes of the file say length " + isize + " (mod 2^32) and CRC-32 " + tcrc.ToString("x8") + ": a second gzip member or bytes behind the stream"); }
                if (res.ArchiveBytes != new FileInfo(archive).Length) { res.Truncated = true; res.Add("archive read incompletely", res.ArchiveBytes + " of " + new FileInfo(archive).Length + " bytes"); }
            }
        }
        catch (Exception e)
        {
            res.Truncated = true;
            res.Add("archive error (" + e.GetType().Name + ")", e.Message);
        }
        res.Seconds = sw.Elapsed.TotalSeconds;
        return res;
    }

    public static ArchiveResult[] CheckAll(string[] archives, string[] roots, string[][] onlys, int parallel)
    {
        ArchiveResult[] results = new ArchiveResult[archives.Length];
        ParallelOptions po = new ParallelOptions();
        po.MaxDegreeOfParallelism = Math.Max(1, parallel);
        Parallel.For(0, archives.Length, po, delegate (int i) { results[i] = Check(archives[i], roots[i], onlys[i]); });
        return results;
    }
}
'@
Add-Type -TypeDefinition $src -Language CSharp

function Invoke-Check([string[]]$archives, [string[]]$roots, [object[]]$onlys, [int]$par, [string]$outDir) {
    $o = New-Object 'string[][]' $archives.Length
    for ($i = 0; $i -lt $archives.Length; $i++) { $o[$i] = [string[]]@($onlys[$i]) }
    $res = [ArchiveCheck]::CheckAll($archives, $roots, $o, $par)
    $script:bad = 0; $script:partial = 0                           # the JSON lines are the output of this function; the counts are not
    foreach ($r in $res) {
        $r.Json()
        if ($r.Differences -gt 0) { $script:bad++ } elseif ($r.SkippedMembers -gt 0) { $script:partial++ }
        if ($outDir) {
            New-Item -ItemType Directory -Force -Path $outDir | Out-Null
            [IO.File]::WriteAllText((Join-Path $outDir ($r.Archive + '.json')), $r.Json() + "`n")
            [IO.File]::WriteAllLines((Join-Path $outDir ($r.Archive + '.problems.txt')), [string[]]$r.Problems)
        }
    }
}

if ($SelfTest) {
    $t = Join-Path ([IO.Path]::GetTempPath()) ('verify_archive_selftest_' + [guid]::NewGuid().ToString('N'))
    $gnu = 'C:\Program Files\Git\usr\bin\tar.exe'
    $bsd = Join-Path $env:SystemRoot 'System32\tar.exe'
    try {
        $s = Join-Path $t 'src'
        New-Item -ItemType Directory -Force -Path (Join-Path $s 'samples\CAM_A'), (Join-Path $s 'samples\CAM_B'), (Join-Path $s 'sweeps\CAM_A') | Out-Null
        $rnd = New-Object Random 7
        function Put([string]$rel, [int]$n) { $b = New-Object byte[] $n; $rnd.NextBytes($b); [IO.File]::WriteAllBytes((Join-Path $s $rel), $b) }
        Put 'samples\CAM_A\a0.jpg' 0; Put 'samples\CAM_A\a1.jpg' 1; Put 'samples\CAM_A\a511.jpg' 511; Put 'samples\CAM_A\a512.jpg' 512; Put 'samples\CAM_A\a513.jpg' 513
        Put 'samples\CAM_A\big.jpg' 3145731; Put 'samples\CAM_B\b.jpg' 4000; Put 'sweeps\CAM_A\s.jpg' 70000
        $long = 'x' * 130 + '.jpg'; Put ('sweeps\CAM_A\' + $long) 900                        # a name longer than 100 characters (GNU long name / pax path)
        $formats = @(@('bsdtar', 9), @('gnu', 9), @('pax', 9), @('ustar', 8))          # name, regular files in the archive (ustar cannot hold the long name: excluded)
        foreach ($f in $formats) {
            $name = $f[0]; $expect = $f[1]; $arc = Join-Path $t ($name + '.tgz'); $dest = Join-Path $t ('dest_' + $name)
            New-Item -ItemType Directory -Force -Path $dest | Out-Null
            $prev = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
            if ($name -eq 'bsdtar') { $msg = & $bsd -czf $arc -C $s samples sweeps 2>&1; $rc = $LASTEXITCODE }
            else {
                $plain = Join-Path $t ($name + '.tar')                                  # GNU tar (Git for Windows) makes the .tar, gzip is done here
                $extra = @(); if ($name -eq 'ustar') { $extra = @('--exclude=xxxxxxxx*') }
                $msg = & $gnu --force-local -cf $plain "--format=$name" @extra -C $s samples sweeps 2>&1; $rc = $LASTEXITCODE
                if ($rc -eq 0) {
                    $in = [IO.File]::OpenRead($plain); $out = [IO.File]::Create($arc); $gz = New-Object IO.Compression.GZipStream($out, [IO.Compression.CompressionMode]::Compress)
                    $in.CopyTo($gz); $gz.Close(); $out.Close(); $in.Close()
                }
            }
            $ErrorActionPreference = $prev
            if ($rc -ne 0) { throw "selftest: $name could not make the archive: $msg" }
            Copy-Item -Recurse -Force (Join-Path $s '*') $dest
            $r = [ArchiveCheck]::Check($arc, $dest, $null)
            if ($r.Differences -ne 0 -or $r.Files -ne $expect) { throw "selftest FAILED ($name): a clean copy must give 0 differences and $expect files: " + $r.Json() + ' ' + ($r.Problems -join '; ') }
            $want = (Get-FileHash -Algorithm SHA256 -LiteralPath $arc).Hash.ToLower()
            if ($r.Sha256 -ne $want -or $r.ArchiveBytes -ne (Get-Item -LiteralPath $arc).Length) { throw "selftest FAILED ($name): the SHA-256 / size of the archive file must be the file's own: " + $r.Json() + " want $want" }
            $r2 = [ArchiveCheck]::Check($arc, $dest, [string[]]@('samples/CAM_A/'))
            if ($r2.Differences -ne 0 -or $r2.Files -ne 6 -or $r2.SkippedMembers -ne ($expect - 6)) { throw "selftest FAILED ($name): -Only samples/CAM_A/ must compare 6 files and skip " + ($expect - 6) + ': ' + $r2.Json() }
            # tamper: one byte, one file missing, one file longer, one file shorter; the long-named file stays intact
            $p = Join-Path $dest 'samples\CAM_A\big.jpg'; $bytes = [IO.File]::ReadAllBytes($p); $bytes[1000000] = [byte](($bytes[1000000] + 1) % 256); [IO.File]::WriteAllBytes($p, $bytes)
            Remove-Item (Join-Path $dest 'samples\CAM_B\b.jpg')
            Add-Content -Path (Join-Path $dest 'sweeps\CAM_A\s.jpg') -Value 'x' -NoNewline
            $q = Join-Path $dest 'samples\CAM_A\a513.jpg'; $qb = [IO.File]::ReadAllBytes($q); [IO.File]::WriteAllBytes($q, $qb[0..511])
            $r3 = [ArchiveCheck]::Check($arc, $dest, $null)
            if ($r3.ContentDiff -ne 1 -or $r3.Missing -ne 1 -or $r3.SizeDiff -ne 2 -or $r3.Differences -ne 4) { throw "selftest FAILED ($name): tampering not found as 1 content + 1 missing + 2 size: " + $r3.Json() + ' ' + ($r3.Problems -join '; ') }
            $cut = Join-Path $t ($name + '_cut.tgz'); $all = [IO.File]::ReadAllBytes($arc); [IO.File]::WriteAllBytes($cut, $all[0..([int]($all.Length / 2))])
            $r4 = [ArchiveCheck]::Check($cut, $dest, $null)
            if (-not $r4.Truncated -or $r4.Differences -lt 1) { throw "selftest FAILED ($name): a truncated archive must be a difference: " + $r4.Json() }
            Write-Host "selftest ($name): clean = 0 differences / $expect files, -Only skips, tampering = 1 content + 1 missing + 2 size, truncated archive flagged"
        }
        # a leading ./ in the member names, an -Only that matches nothing, an archive without the end-of-archive marker
        $dot = Join-Path $t 'dot.tgz'; $dd = Join-Path $t 'dest_dot'; New-Item -ItemType Directory -Force -Path $dd | Out-Null; Copy-Item -Recurse -Force (Join-Path $s '*') $dd
        & $bsd -czf $dot -C $s ./samples ./sweeps                                    # member names start with ./
        $rd = [ArchiveCheck]::Check($dot, $dd, $null)
        if ($rd.Differences -ne 0 -or $rd.Files -ne 9) { throw "selftest FAILED (./ names): " + $rd.Json() + ' ' + ($rd.Problems -join '; ') }
        $rd2 = [ArchiveCheck]::Check($dot, $dd, [string[]]@('samples/CAM_A/'))
        if ($rd2.Differences -ne 0 -or $rd2.Files -ne 6) { throw "selftest FAILED (./ names with -Only must still match): " + $rd2.Json() }
        $rn = [ArchiveCheck]::Check($dot, $dd, [string[]]@('nothing/here/'))
        if (-not $rn.NothingCompared -or $rn.Differences -ne 1) { throw "selftest FAILED: an -Only that matches nothing must be a difference: " + $rn.Json() }
        $plain = Join-Path $t 'noend.tar'; & $bsd -cf $plain -C $s samples sweeps
        $pb = [IO.File]::ReadAllBytes($plain); $end = $pb.Length; while ($end -gt 0 -and $pb[$end - 1] -eq 0) { $end-- }
        $end = [int]([math]::Ceiling($end / 512.0) * 512)                                           # cut exactly at a block boundary, behind the last member's padding
        $noend = Join-Path $t 'noend.tgz'; $o = [IO.File]::Create($noend); $gz = New-Object IO.Compression.GZipStream($o, [IO.Compression.CompressionMode]::Compress); $gz.Write($pb, 0, $end); $gz.Close(); $o.Close()
        $re = [ArchiveCheck]::Check($noend, $dd, $null)
        if (-not $re.Truncated -or $re.Differences -lt 1 -or ($re.Problems -join ';') -notmatch 'end-of-archive marker') { throw "selftest FAILED: an archive without the end-of-archive marker must be a difference (and say so): " + $re.Json() + ' ' + ($re.Problems -join '; ') }
        Write-Host 'selftest (names with ./, -Only that matches nothing, archive without the end marker): handled'
        # ---- crafted archives (a small tar writer): structure, tampering at chunk borders, gzip oddities, member types, names
        function New-TarEntry {
            param([string]$Name, [byte[]]$Data = (New-Object byte[] 0), [string]$Type = '0', $SizeField = $null, [string]$Prefix = '', [string]$SizeText = '', [switch]$BadSum, [switch]$Base256)
            $h = New-Object byte[] 512; $ascii = [Text.Encoding]::ASCII
            $nb = $ascii.GetBytes($Name); [Array]::Copy($nb, 0, $h, 0, [Math]::Min(100, $nb.Length))
            [Array]::Copy($ascii.GetBytes("0000644`0"), 0, $h, 100, 8); [Array]::Copy($ascii.GetBytes("0000000`0"), 0, $h, 108, 8); [Array]::Copy($ascii.GetBytes("0000000`0"), 0, $h, 116, 8)
            $size = if ($null -ne $SizeField) { [long]$SizeField } else { [long]$Data.Length }
            if ($SizeText) { [Array]::Copy($ascii.GetBytes($SizeText.PadRight(12)), 0, $h, 124, 12) }
            elseif ($Base256) { $h[124] = 0x80; for ($i = 11; $i -ge 1; $i--) { $h[124 + $i] = [byte]($size -band 0xFF); $size = $size -shr 8 } }
            else { [Array]::Copy($ascii.GetBytes(([Convert]::ToString($size, 8)).PadLeft(11, '0') + "`0"), 0, $h, 124, 12) }
            [Array]::Copy($ascii.GetBytes("00000000000`0"), 0, $h, 136, 12)
            $h[156] = [byte][char]$Type
            [Array]::Copy($ascii.GetBytes("ustar`000"), 0, $h, 257, 8)
            if ($Prefix) { $pb = $ascii.GetBytes($Prefix); [Array]::Copy($pb, 0, $h, 345, [Math]::Min(155, $pb.Length)) }
            for ($i = 148; $i -lt 156; $i++) { $h[$i] = 32 }
            $sum = 0; foreach ($b in $h) { $sum += $b }
            [Array]::Copy($ascii.GetBytes(([Convert]::ToString($sum, 8)).PadLeft(6, '0') + "`0 "), 0, $h, 148, 8)
            if ($BadSum) { $h[148] = [byte][char]'7'; $h[149] = [byte][char]'7'; $h[150] = [byte][char]'7' }
            $ms = New-Object IO.MemoryStream
            $ms.Write($h, 0, 512)
            if ($Data.Length -gt 0) { $ms.Write($Data, 0, $Data.Length); $pad = (512 - ($Data.Length % 512)) % 512; if ($pad -gt 0) { $ms.Write((New-Object byte[] $pad), 0, $pad) } }
            return , $ms.ToArray()
        }
        function Join-Bytes([object[]]$Parts) { $ms = New-Object IO.MemoryStream; foreach ($p in $Parts) { $ms.Write($p, 0, $p.Length) }; return , $ms.ToArray() }
        function Save-Tgz([byte[]]$Tar, [string]$Path) { $o = [IO.File]::Create($Path); $gz = New-Object IO.Compression.GZipStream($o, [IO.Compression.CompressionMode]::Compress); $gz.Write($Tar, 0, $Tar.Length); $gz.Close(); $o.Close() }
        $END = New-Object byte[] 1024
        $crafted = Join-Path $t 'crafted'; $cd = Join-Path $crafted 'dest'; New-Item -ItemType Directory -Force -Path (Join-Path $cd 'sub\dir') | Out-Null
        $dA = [Text.Encoding]::ASCII.GetBytes('hello world'); $dB = New-Object byte[] 3145731; (New-Object Random 11).NextBytes($dB)
        [IO.File]::WriteAllBytes((Join-Path $cd 'a.bin'), $dA); [IO.File]::WriteAllBytes((Join-Path $cd 'b.bin'), $dB); [IO.File]::WriteAllBytes((Join-Path $cd 'sub\dir\x.bin'), $dA)
        function Chk([string]$Name, [byte[]]$Tar, [string]$Dest = $cd, [string[]]$Only = $null) { $a = Join-Path $crafted ($Name + '.tgz'); Save-Tgz $Tar $a; return [ArchiveCheck]::Check($a, $Dest, $Only) }
        function Expect([string]$What, $Res, [scriptblock]$Ok) { if (-not (& $Ok $Res)) { throw "selftest FAILED (crafted: $What): " + $Res.Json() + ' ' + ($Res.Problems -join '; ') } }
        $eA = New-TarEntry 'a.bin' $dA; $eB = New-TarEntry 'b.bin' $dB
        Expect 'clean' (Chk 'clean' (Join-Bytes @($eA, $eB, $END))) { param($r) $r.Differences -eq 0 -and $r.Files -eq 2 -and $r.Bytes -eq (11 + 3145731) }
        Expect 'ustar prefix' (Chk 'prefix' (Join-Bytes @((New-TarEntry 'x.bin' $dA -Prefix 'sub/dir'), $END))) { param($r) $r.Differences -eq 0 -and $r.Files -eq 1 }
        Expect 'space-padded octal size field' (Chk 'spacepad' (Join-Bytes @((New-TarEntry 'a.bin' $dA -SizeText '         13 '), $END))) { param($r) $r.Differences -eq 0 -and $r.Files -eq 1 }
        Expect 'base-256 size field' (Chk 'b256' (Join-Bytes @((New-TarEntry 'a.bin' $dA -Base256), $END))) { param($r) $r.Differences -eq 0 -and $r.Files -eq 1 }
        $px = [Text.Encoding]::ASCII.GetBytes("11 size=11`n")
        Expect 'pax size override' (Chk 'paxsize' (Join-Bytes @((New-TarEntry 'pax' $px -Type 'x'), (New-TarEntry 'a.bin' -SizeField 0 -Data $dA), $END))) { param($r) $r.Differences -eq 0 -and $r.Files -eq 1 }
        Expect 'unsafe name' (Chk 'unsafe' (Join-Bytes @((New-TarEntry '../evil.bin' $dA), $eA, $END))) { param($r) $r.Unsafe -eq 1 -and $r.Differences -eq 1 }
        Expect 'symlink member' (Chk 'symlink' (Join-Bytes @((New-TarEntry 'ln' (New-Object byte[] 0) -Type '2'), $eA, $END))) { param($r) $r.Other -eq 1 -and $r.Differences -eq 1 }
        Expect 'fifo member' (Chk 'fifo' (Join-Bytes @((New-TarEntry 'ff' (New-Object byte[] 0) -Type '6'), $eA, $END))) { param($r) $r.Other -eq 1 -and $r.Differences -eq 1 }
        Expect 'bad header checksum' (Chk 'badsum' (Join-Bytes @($eA, (New-TarEntry 'b.bin' $dB -BadSum), $END))) { param($r) $r.BadHeader -eq 1 -and $r.Differences -ge 1 }
        Expect 'non-zero data behind the end marker' (Chk 'afterend' (Join-Bytes @($eA, $END, $eB, $END))) { param($r) $r.Truncated -and ($r.Problems -join ';') -match 'end-of-archive marker' }
        Expect 'one zero block only' (Chk 'onezero' (Join-Bytes @($eA, (New-Object byte[] 512)))) { param($r) $r.Truncated -and ($r.Problems -join ';') -match 'end-of-archive marker' }
        Expect 'no end marker' (Chk 'noend2' (Join-Bytes @($eA, $eB))) { param($r) $r.Truncated -and ($r.Problems -join ';') -match 'end-of-archive marker' }
        Expect 'empty tar (0 bytes)' (Chk 'empty' (New-Object byte[] 0)) { param($r) $r.Differences -ge 1 }
        Expect 'tar of zero blocks only' (Chk 'zeros' $END) { param($r) $r.NothingCompared -and $r.Differences -ge 1 }
        $good = Join-Path $crafted 'good.tgz'; Save-Tgz (Join-Bytes @($eA, $END)) $good; $gb = [IO.File]::ReadAllBytes($good)
        function ChkRaw([string]$Name, [byte[]]$Bytes, [string[]]$Only = $null) { $a = Join-Path $crafted ($Name + '.tgz'); [IO.File]::WriteAllBytes($a, $Bytes); return [ArchiveCheck]::Check($a, $cd, $Only) }
        $second = Join-Path $crafted 'second.tgz'; Save-Tgz (Join-Bytes @((New-TarEntry 'not-on-disk.bin' $dA), $END)) $second
        Expect 'a second gzip member with another tar' (ChkRaw 'multi' ([byte[]]($gb + [IO.File]::ReadAllBytes($second)))) { param($r) $r.Truncated -and ($r.Problems -join ';') -match 'gzip trailer' }
        Expect 'random bytes behind the gzip stream' (ChkRaw 'garbage' ([byte[]]($gb + (1..100 | ForEach-Object { [byte]($_ * 7 % 251 + 1) })))) { param($r) $r.Truncated }
        Expect 'zero bytes behind the gzip stream' (ChkRaw 'zeropad' ([byte[]]($gb + (New-Object byte[] 64)))) { param($r) $r.Truncated }
        Expect 'gzip trailer cut' (ChkRaw 'cut8' ([byte[]]($gb[0..($gb.Length - 9)]))) { param($r) $r.Truncated }
        Expect 'empty file' (ChkRaw 'zerofile' (New-Object byte[] 0)) { param($r) $r.Differences -ge 1 }
        # tampering at the borders of the 1 MiB compare chunks, in both directions
        foreach ($off in 0, 1048575, 1048576, 2097151, 2097152, 3145730) {
            foreach ($delta in 1, -1) {
                $v = $dB.Clone(); $v[$off] = [byte](($v[$off] + $delta + 256) % 256)
                $vd = Join-Path $crafted 'dest_tamper'; New-Item -ItemType Directory -Force -Path $vd | Out-Null; [IO.File]::WriteAllBytes((Join-Path $vd 'b.bin'), $v)
                Expect "byte $off changed by $delta" (Chk 'tamper' (Join-Bytes @($eB, $END)) $vd) { param($r) $r.ContentDiff -eq 1 -and $r.Differences -eq 1 }
            }
        }
        # a locked file and a folder where the archive has a file
        $lk = Join-Path $crafted 'dest_lock'; New-Item -ItemType Directory -Force -Path $lk | Out-Null; [IO.File]::WriteAllBytes((Join-Path $lk 'a.bin'), $dA)
        $fh = [IO.File]::Open((Join-Path $lk 'a.bin'), 'Open', 'Read', 'None')
        try { Expect 'a file locked by another process' (Chk 'locked' (Join-Bytes @($eA, $END)) $lk) { param($r) $r.Differences -ge 1 } } finally { $fh.Close() }
        $dr = Join-Path $crafted 'dest_dir'; New-Item -ItemType Directory -Force -Path (Join-Path $dr 'a.bin') | Out-Null
        Expect 'a folder where the archive has a file' (Chk 'isdir' (Join-Bytes @($eA, $END)) $dr) { param($r) $r.Differences -ge 1 }
        Write-Host 'selftest (crafted archives): ustar prefix, base-256 and pax sizes, unsafe names, links / fifo, bad checksum, data behind the end marker, no / one end block, empty files, second gzip member, bytes behind the gzip stream, cut trailer, tampering at chunk borders in both directions, locked file, folder instead of file'
        # the command line: a comma list of prefixes (one string from bash), the exit code 3 of a partial comparison, exit 2 for no arguments
        $pa = Join-Path $crafted 'partial.tgz'; Save-Tgz (Join-Bytes @($eA, $eB, $END)) $pa
        $o3 = & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $PSCommandPath -Archive $pa -Dest $cd -Only 'a.bin,nothing/'; $rc3 = $LASTEXITCODE
        if ($rc3 -ne 3 -or ($o3 -join '') -notmatch '"files": 1' -or ($o3 -join '') -notmatch '"skipped_members": 1' -or ($o3 -join '') -notmatch '"differences": 0') { throw "selftest FAILED (command line, comma list of -Only, partial): exit $rc3, output: $o3" }
        $o4 = & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $PSCommandPath; $rc4 = $LASTEXITCODE
        if ($rc4 -ne 2) { throw "selftest FAILED (command line, no arguments must exit 2): exit $rc4" }
        Write-Host 'selftest (command line): comma list of -Only, exit 3 for a partial comparison, exit 2 for no arguments'
        # the command line: the JSON line is printed, the exit code says whether every archive is clean, -List reads archive|dest|prefixes
        $cli = Join-Path $t 'cli.tgz'; $cd = Join-Path $t 'dest_cli'; New-Item -ItemType Directory -Force -Path $cd | Out-Null
        & $bsd -czf $cli -C $s samples sweeps; Copy-Item -Recurse -Force (Join-Path $s '*') $cd
        $o1 = & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $PSCommandPath -Archive $cli -Dest $cd; $rc1 = $LASTEXITCODE
        if ($rc1 -ne 0 -or ($o1 -join '') -notmatch '"differences": 0' -or ($o1 -join '') -notmatch '"archive_sha256": "[0-9a-f]{64}"') { throw "selftest FAILED (command line, clean): exit $rc1, output: $o1" }
        Remove-Item (Join-Path $cd 'sweeps\CAM_A\s.jpg')
        $lst = Join-Path $t 'list.txt'; [IO.File]::WriteAllText($lst, "$cli|$cd`n$cli|$cd|samples/CAM_A/;samples/CAM_B/`n")
        $o2 = & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $PSCommandPath -List $lst -Parallel 2; $rc2 = $LASTEXITCODE
        if ($rc2 -ne 1 -or @($o2).Count -ne 2 -or $o2[0] -notmatch '"missing": 1' -or $o2[1] -notmatch '"differences": 0' -or $o2[1] -notmatch '"skipped_members": 2' -or $o2[1] -notmatch '"files": 7') { throw "selftest FAILED (command line, -List with one bad archive): exit $rc2, output: $o2" }
        Write-Host 'selftest (command line): JSON line printed, exit 0 when clean, exit 1 and the difference shown when not, -List with prefixes'
        Write-Host 'selftest ok'
        exit 0
    }
    finally { Remove-Item -Recurse -Force $t -ErrorAction SilentlyContinue }
}

$archives = @(); $roots = @(); $onlys = @()
if ($List) {
    foreach ($line in Get-Content -LiteralPath $List) {
        if (-not $line.Trim()) { continue }
        $f = $line.Split('|')
        $archives += $f[0]; $roots += $f[1]
        $onlys += , $(if ($f.Length -gt 2 -and $f[2]) { [string[]]$f[2].Split(';') } else { $null })
    }
} elseif ($Archive -and $Dest) {
    $archives = @($Archive); $roots = @($Dest); $onlys = @(, $Only)
} else {
    Write-Host 'usage: verify_archive.ps1 -Archive <archive> -Dest <folder> [-Only prefix,...] [-OutDir d] | -List <file> [-Parallel n] [-OutDir d] | -SelfTest'
    exit 2
}
$script:bad = 0; $script:partial = 0
Invoke-Check $archives $roots $onlys $Parallel $OutDir
exit $(if ($script:bad -gt 0) { 1 } elseif ($script:partial -gt 0) { 3 } else { 0 })
