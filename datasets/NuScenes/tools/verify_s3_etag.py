"""One-off: check a downloaded file against its S3 ETag (plain md5 or multipart md5-of-md5s "<hex>-<parts>").

wsl python3 datasets/NuScenes/tools/verify_s3_etag.py <file> <etag>   -> exit 0 if it matches
"""
import hashlib
import os
import sys

path, etag = sys.argv[1], sys.argv[2].strip('"')
size = os.path.getsize(path)
if "-" not in etag:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    ok = h.hexdigest() == etag
else:
    want, nparts = etag.split("-")
    nparts = int(nparts)
    # part size: the smallest whole MiB that gives nparts parts
    mib = 1 << 20
    part = -(-size // nparts)
    part = -(-part // mib) * mib
    if -(-size // part) != nparts:
        sys.exit(f"cannot infer the part size for {nparts} parts of {size} bytes")
    digests = []
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(part), b""):
            digests.append(hashlib.md5(chunk).digest())
    ok = hashlib.md5(b"".join(digests)).hexdigest() == want and len(digests) == nparts
print(("OK " if ok else "MISMATCH ") + os.path.basename(path))
sys.exit(0 if ok else 1)
