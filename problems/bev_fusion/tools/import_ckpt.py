"""The step `import` of a BEV run that trains nothing (a measurement of an existing network): the newest checkpoint of the run folder named by the
config's "_import" (relative to the problem's data folder, e.g. "runs/t21_ctl_s{seed}") is copied, byte for byte, into this run's folder, where the
evaluation step reads it. A copy that is there and equal is kept; one that differs is refused.

    python tools/import_ckpt.py <the run's config.json>"""
import glob
import hashlib
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "net"))
import netpath  # noqa: E402


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv):
    cfg = json.load(open(argv[0], encoding="utf-8"))
    src_dir = os.path.join(netpath.SITE, cfg["_import"].format(seed=cfg["seed"]))
    found = sorted(glob.glob(os.path.join(src_dir, "ckpt_*.pt")))
    if not found:
        raise SystemExit(f"no checkpoint in {src_dir}")
    src = found[-1]
    dst = os.path.join(os.path.dirname(os.path.abspath(argv[0])), os.path.basename(src))
    if not os.path.isfile(dst):
        shutil.copyfile(src, dst + ".part")
        os.replace(dst + ".part", dst)
    if sha256(src) != sha256(dst):
        raise SystemExit(f"{dst} is not {src}: move it away")
    print(f"imported {src} ({os.path.getsize(dst)} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
