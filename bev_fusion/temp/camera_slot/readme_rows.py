"""Scratch: apply the three camera-slot README row edits to the working tree and to the HEAD blobs
(the blobs go into a temporary index, so the other chats' uncommitted README edits stay out of the commit).
Usage: python3 readme_rows.py <repo root>   -> writes temp/camera_slot/blob_*.md next to this file."""
import subprocess
import sys
from pathlib import Path

root = Path(sys.argv[1])
here = Path(__file__).resolve().parent
EDITS = {
    "bev_fusion/training/README.md": [
        ("`CachedFrontRadarDataset` (no image is read), `collate_cached`,",
         "`CachedFrontRadarDataset` (no image is read; one cache per feature source of the `CAMERAS` table of `model.py`), `collate_cached`,"),
        ("lift `pd` / `p` / `s` (LSS, fallback), camera temporal",
         "camera feature sources = the `CAMERAS` table (`dualbev`, `bevfusion`, `dual` = both lifted separately and merged; a new source is one table entry), lift `pd` / `p` / `s` (LSS, fallback), camera temporal"),
    ],
    "bev_fusion/tests/README.md": [
        ("refusals (lift s, unknown value) | camera BEV equality / shape, flags | passes on CPU (2026-10-30) |",
         "refusals (lift s, unknown value); the dual branches (shapes, branch select, gradients to both reductions and the merge); the `CAMERAS` table (names, checkpoint keys and registration order, dataset items and collate per source, refusals) | camera BEV equality / shape, flags, state_dict keys | passes on CPU (2026-11-01) |"),
    ],
}
for rel, pairs in EDITS.items():
    work = root / rel
    cur = work.read_bytes().decode("utf-8")
    head = subprocess.run(["git", "show", "HEAD:" + rel], cwd=root, capture_output=True, check=True).stdout.decode("utf-8")
    for old, new in pairs:
        assert cur.count(old) == 1, (rel, "working tree", old[:40], cur.count(old))
        assert head.count(old) == 1, (rel, "HEAD", old[:40], head.count(old))
        cur = cur.replace(old, new)
        head = head.replace(old, new)
    work.write_bytes(cur.encode("utf-8"))
    (here / ("blob_" + Path(rel).parent.name + "_" + Path(rel).name)).write_bytes(head.encode("utf-8"))
    print("ok", rel, len(pairs), "edits")
