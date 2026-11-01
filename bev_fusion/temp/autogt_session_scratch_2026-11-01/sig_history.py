"""scratch: the generator signature at every commit that changed the generator (for autogt/versions.csv).
Run in WSL from C:\Auto-research (repository root):  python3 bev_fusion/temp/autogt_session_scratch_2026-11-01/sig_history.py
For each commit: git archive of the generator folder into a temp dir, then a fresh python process computes NuScenesGT.signature with default options."""
import os
import subprocess
import sys
import tempfile
import textwrap

REPO = os.getcwd()
PY = os.path.expanduser("~/miniconda3/envs/gcar/bin/python")
SUITE = ["354caef", "769aafc", "a49bbd3", "79c916f"]
PRE = ["7f5cb10", "7c8f12e", "8612a56"]                       # training/gt_bev.py era (signature in the cache key since 7f5cb10)

SNIP_SUITE = textwrap.dedent("""
    import sys, types
    sys.path.insert(0, sys.argv[1])
    from autogt.nuscenes_gt import NuScenesGT
    opt = dict(stationary_fix=True, map_offsets={}, use_flow=True, use_radar=True, flow_config=None, boxes_at="camera")
    print(NuScenesGT.signature(types.SimpleNamespace(**opt)))
""")
SNIP_PRE = textwrap.dedent("""
    import sys, types
    sys.path.insert(0, sys.argv[1])
    import gt_bev
    opt = dict(stationary_fix=True, map_offsets={}, use_flow=True, use_radar=True, flow_config=None, boxes_at="camera")
    print(gt_bev.NuScenesGT.signature(types.SimpleNamespace(**opt)))
""")


def at(commit, folder, snippet):
    with tempfile.TemporaryDirectory() as d:
        p = subprocess.run(f"git archive {commit} {folder} | tar -x -C {d}", shell=True, cwd=REPO, capture_output=True, text=True)
        if p.returncode:
            return "no folder at this commit: " + p.stderr.strip()[:80]
        root = os.path.join(d, folder) if folder.endswith("training") else d
        if folder.endswith("autogt"):
            root = os.path.join(d, os.path.dirname(folder))           # the folder that CONTAINS the package
        r = subprocess.run([PY, "-c", snippet, root], capture_output=True, text=True, cwd=d)
        return (r.stdout.strip() or r.stderr.strip().splitlines()[-1][:120])


for c in SUITE:
    folder = "autogt" if c == "79c916f" else "bev_fusion/autogt"
    print("suite", c, subprocess.run(["git", "log", "-1", "--format=%ad", "--date=format:%m-%d %H:%M", c], capture_output=True, text=True, cwd=REPO).stdout.strip(),
          at(c, folder, SNIP_SUITE))
for c in PRE:
    print("pre  ", c, subprocess.run(["git", "log", "-1", "--format=%ad", "--date=format:%m-%d %H:%M", c], capture_output=True, text=True, cwd=REPO).stdout.strip(),
          at(c, "bev_fusion/training", SNIP_PRE))
print("known stores: 962e788566 (pre-suite), 7fe1149566, fefd519974, 15efbde2d1")
