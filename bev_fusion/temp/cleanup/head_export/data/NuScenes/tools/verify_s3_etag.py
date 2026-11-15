"""Check a downloaded file against its S3 ETag: a plain md5, or the multipart form "<md5 of the part md5s>-<number of parts>".

    python3 data/NuScenes/tools/verify_s3_etag.py <file> <etag> [--part-size MiB]   -> exit 0 if it matches
    python3 data/NuScenes/tools/verify_s3_etag.py --selftest

A multipart ETag does not say how big the parts were, and the file size and the part count do not give it (20 MiB in 3 parts can be
7, 8, 9 ... MiB parts). Without --part-size the usual sizes are tried (8 MiB: aws cli and boto3, 5 MiB: the minimum, 15 MiB: s3cmd, 16, 32,
64, 100, 128 ... MiB), only those that give the part count in the ETag; the first that matches wins and is printed. If none matches the
upload may have used other sizes (or parts of different sizes): give --part-size, or the file is damaged.
"""
import hashlib
import os
import sys
import tempfile

MiB = 1 << 20
USUAL_MIB = (8, 5, 15, 16, 10, 32, 64, 100, 128, 25, 50, 256, 512, 1024)


def md5_of(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def multipart_etag(path, part_bytes):
    """"<md5 of the concatenated md5 digests of the parts>-<number of parts>" for parts of part_bytes"""
    digests = []
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(part_bytes), b""):
            digests.append(hashlib.md5(chunk).digest())
    return f"{hashlib.md5(b''.join(digests)).hexdigest()}-{len(digests)}"


def candidate_parts(size, nparts, part_mib=None):
    """part sizes in MiB that give nparts parts for a file of `size` bytes: the given one, or the usual ones"""
    cands = [part_mib] if part_mib else list(USUAL_MIB)
    return [m for m in cands if -(-size // (m * MiB)) == nparts]


def verify(path, etag, part_mib=None):
    """-> (ok, text); text says which part size matched, or what was tried"""
    etag = etag.strip('"')
    size = os.path.getsize(path)
    if "-" not in etag:
        return md5_of(path) == etag, "plain md5"
    want, n = etag.split("-")
    n = int(n)
    cands = candidate_parts(size, n, part_mib)
    if not cands:
        lo, hi = (size / n / MiB, size / (n - 1) / MiB) if n > 1 else (size / MiB, float("inf"))
        what = f"{part_mib} MiB" if part_mib else "the usual sizes " + ", ".join(map(str, USUAL_MIB)) + " MiB"
        return False, f"{what} do not give {n} parts for {size} bytes (the part size would be between {lo:.2f} and {hi:.2f} MiB): give --part-size"
    for m in cands:
        if multipart_etag(path, m * MiB) == f"{want}-{n}":
            return True, f"{n} parts of {m} MiB"
    return False, f"no match with parts of {', '.join(str(m) for m in cands)} MiB"


def _selftest():
    rng = os.urandom
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "blob")
        with open(p, "wb") as f:                                           # 20 MiB + 123 bytes: 3 parts of 8 MiB (the old guess was 7 MiB)
            f.write(rng(20 * MiB + 123))
        e8, e5 = multipart_etag(p, 8 * MiB), multipart_etag(p, 5 * MiB)
        assert e8.endswith("-3") and e5.endswith("-5"), (e8, e5)
        assert verify(p, e8) == (True, "3 parts of 8 MiB") and verify(p, '"%s"' % e5) == (True, "5 parts of 5 MiB")
        assert verify(p, md5_of(p)) == (True, "plain md5")
        assert verify(p, e8, part_mib=8)[0] and not verify(p, e8, part_mib=5)[0]                    # a given size is the only one tried
        ok, text = verify(p, e8.split("-")[0] + "-4")                                              # 4 parts: no usual size gives them
        assert not ok and "--part-size" in text, text
        ok, text = verify(p, "0" * 32 + "-3")                                                      # the right count, the wrong digest
        assert not ok and "no match" in text, text
        with open(p, "r+b") as f:                                                                  # a damaged byte
            f.seek(9 * MiB)
            f.write(b"\x00\x01\x02")
        assert not verify(p, e8)[0] and not verify(p, md5_of(p) + "0")[0]
    print("verify_s3_etag selftest ok: plain md5, multipart with 8 and 5 MiB parts (the 20 MiB / 3 parts case the old inference missed), a given size, "
          "an impossible part count, a wrong digest, a damaged file")


if __name__ == "__main__":
    args = sys.argv[1:]
    if args == ["--selftest"]:
        _selftest()
        sys.exit(0)
    part = None
    if "--part-size" in args:
        i = args.index("--part-size")
        part = int(args[i + 1])
        del args[i:i + 2]
    if len(args) != 2:
        sys.exit(__doc__)
    ok, text = verify(args[0], args[1], part)
    print(("OK " if ok else "MISMATCH ") + os.path.basename(args[0]) + f" ({text})")
    sys.exit(0 if ok else 1)
