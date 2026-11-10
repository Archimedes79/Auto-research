"""Scratch: build blobs of shared files = HEAD version + my rows (inserted after the first line that starts with an anchor), never the working-tree copy.
    python3 add_rows.py   -> writes temp/cleanup/blobs/<name> and prints what each blob adds against HEAD"""
import difflib
import os
import subprocess

REPO = "/mnt/c/Auto-research"
OUT = os.path.join(REPO, "bev_fusion/temp/cleanup/blobs")
ROWS = {
    "bev_fusion/evaluation/README.md": (
        "| `bev_eval.py` + `evalsuite.json` |",
        ["| `bev_free.py` | the BEV network run WITHOUT the ground truth (D72): `free_source(built_source)` (a Source whose items carry no `gt`, `Sample(reference=None)`, `describe()` without a ground-truth signature, the planned frame's dataset key checked) and `FreeSubject(built_subject)` (the model of a `BEVSubject` called the way `train.forward_loss` calls it before the loss: ego gap, host copies, autocast, carried state; `info` without `loss`); the outputs are built by the same functions as `BEVSubject.predict` builds them and are bit-equal to it; plugged into `Evaluation` through the adapter functions `free_source` / `free_subject` of `bev_eval.py`. The output-building lines are copied from the frozen `bev_adapter.py`: the test fails when they drift | `../tests/test_bev_free.py` |"]),
    "bev_fusion/tests/README.md": (
        "| `test_evaluation_api.py` |",
        ["| `test_bev_free.py` | `evaluation/bev_free.py` on REAL frames (default `runs/t20_base_s0`), CPU, the first 4 frames of val scene-0003, minutes and about 6 GB, not in `run_gates.sh`: the live pair (ground truth read, loss computed) against `FreeSource` / `FreeSubject`: the outputs of every dt and the taps bit-equal, no reference and no `gt` in the sample, no loss in `info`, the same frames and dataset keys, the memory carried over the 4 frames and a second pass after `reset()` equal, the memory confidence empty at the first frame and growing, a wrong dataset item and a broken input refused with `ValueError` (never `SystemExit`) |"]),
}


def head(path):
    return subprocess.run(["git", "-C", REPO, "show", f"HEAD:{path}"], capture_output=True, text=True, check=True).stdout


def main():
    os.makedirs(OUT, exist_ok=True)
    for path, (anchor, rows) in ROWS.items():
        lines = head(path).split("\n")
        i = next(k for k, l in enumerate(lines) if l.startswith(anchor))
        new = lines[:i + 1] + rows + lines[i + 1:]
        dest = os.path.join(OUT, path.replace("/", "_"))
        with open(dest, "w", encoding="utf-8", newline="") as f:
            f.write("\n".join(new))
        diff = [l for l in difflib.unified_diff(lines, new, lineterm="", n=0) if l[:1] in "+-" and l[:3] not in ("+++", "---")]
        print(f"{path}: +{sum(l[0] == '+' for l in diff)} -{sum(l[0] == '-' for l in diff)} -> {dest}")


main()
