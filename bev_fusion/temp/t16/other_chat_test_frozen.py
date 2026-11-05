"""The frozen frame of bev_fusion (training/FROZEN.md): the files of training/frozen_files.txt equal their version at the freeze tag (bf/frame-v1) in HEAD and in the working tree.
A difference is listed and fails the test; a bug fix to a frozen file moves the tag (FROZEN.md says how). Without the tag (a clone that did not fetch tags) the test says so and passes.
Run from the repository root or bev_fusion: python tests/test_frozen.py"""
import os
import subprocess
import sys

TAG = "bf/frame-v1"
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=HERE, capture_output=True, text=True, check=True).stdout.strip()


def git(*a):
    return subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True)


def main():
    if git("rev-parse", "--verify", "--quiet", TAG + "^{commit}").returncode != 0:
        print(f"test_frozen: no tag {TAG} here (git fetch --tags): nothing to compare, passes")
        return 0
    paths = [ln.strip() for ln in open(os.path.join(REPO, "bev_fusion", "training", "frozen_files.txt"), encoding="utf-8") if ln.strip() and not ln.startswith("#")]
    changed = [p for p in paths if git("diff", "--quiet", TAG, "--", p).returncode != 0]
    missing = [p for p in paths if not os.path.exists(os.path.join(REPO, p))]
    for p in changed:
        print(f"  FROZEN FILE CHANGED since {TAG}: {p}")
    for p in missing:
        print(f"  FROZEN FILE MISSING: {p}")
    if changed or missing:
        print("test_frozen FAILED: a frozen file changes only for a bug fix, with its test and a critic pass, and the tag moves (training/FROZEN.md)")
        return 1
    print(f"test_frozen passed: {len(paths)} frozen files equal {TAG}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
