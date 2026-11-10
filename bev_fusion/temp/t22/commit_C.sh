#!/bin/bash
# scratch: commit C: the test file (autocast case, literal marker keys, docstring), the corrected numbers in the FROZEN.md line (text only: FROZEN.md is not in frozen.json) and my tests/README.md row, rebuilt from the CURRENT HEAD
cd /mnt/c/Auto-research || exit 1
python3 - <<'EOF'
import subprocess
path = "bev_fusion/tests/README.md"
old = "15 more planted bugs killed (scratch `temp/t22/mut_K2.py`); the test removes"
new = ("'all' under autocast (bf16 on the CPU: the current-frame student call has to come before the older-frame no_grad call, the swapped order is caught); 16 more planted bugs killed "
       "(scratch `temp/t22/mut_K2.py`, `mut_K5.py`, `mut_K4.sh`; the camera-control and marker-key mutants were re-run after a first run died of a missing package in the private copy); the test removes")
head = subprocess.run(["git", "show", "HEAD:" + path], capture_output=True, check=True).stdout.decode("utf-8")
assert head.count(old) == 1, head.count(old)
open("bev_fusion/temp/t22/blob_tests_README.md", "w", encoding="utf-8", newline="").write(head.replace(old, new))
print("README copy rebuilt from HEAD", subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip())
EOF
P="bev_fusion/tests/test_student_chain.py bev_fusion/FROZEN.md"
python3 autoresearch/tools/check_git.py --paths $P bev_fusion/tests/README.md 2>&1 | tail -4
cat > bev_fusion/temp/t22/commit_msg_C.txt <<'EOF'
t22 tests: autocast order case, literal BUILT.json keys, stale docstring; FROZEN.md numbers (159 checks, 44 planted bugs)

Review chat's small points: the docstring of test_student_chain no longer says 'all' = NotImplementedError; the autocast ordering comment of feature_cache.camera_bevs has a test (CPU bf16, training mode, 'all': a gradient on every trainable student tensor; the swapped call order is caught by it). My own verification gap closed: the marker-key test compared BUILT.json keys with older_key() itself, it now compares literal strings (a key function that drops the stride is caught), and the five mutants whose first run died of a missing package in the private copy were re-run on a complete copy (all killed). FROZEN.md line: 159 checks, 44 planted bugs. The README row of the test says so. Open, not done: the BUILT.json marker does not record the scene set (a cache built for a small split satisfies the check for a bigger one); it needs a frozen-file change (startable.py) and is not needed while t22_B2_s0 is built from its own config.
EOF
python3 autoresearch/tools/commit_paths.py -F bev_fusion/temp/t22/commit_msg_C.txt --trailer "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>" $P \
  --blob bev_fusion/tests/README.md=bev_fusion/temp/t22/blob_tests_README.md 2>&1 | tail -6
