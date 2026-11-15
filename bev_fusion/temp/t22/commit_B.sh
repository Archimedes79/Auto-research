#!/bin/bash
# scratch: commit B of the t22 frozen-frame acceptance: after commit A the frozen files in the working tree ARE the committed ones; test_frozen --update rewrites frozen.json, then FROZEN.md + frozen.json are committed through a temporary index
cd /mnt/c/Auto-research/bev_fusion || exit 1
PY=$HOME/miniconda3/envs/gcar/bin/python
export CUDA_VISIBLE_DEVICES=""
# the working copies of all frozen files must equal HEAD now (no foreign uncommitted edit would be hashed into frozen.json)
dirty=$($PY - <<'EOF'
import json, subprocess
files = list(json.load(open("frozen.json"))["files"] if "files" in json.load(open("frozen.json")) else json.load(open("frozen.json")))
out = subprocess.run(["git", "status", "--short", "--"] + ["bev_fusion/" + f for f in files if "/" in f], cwd="..", capture_output=True, text=True).stdout
print(out.strip())
EOF
)
if [ -n "$dirty" ]; then echo "FROZEN FILES NOT CLEAN (stop):"; echo "$dirty"; exit 1; fi
echo "frozen files equal HEAD: ok"
cp frozen.json temp/t22/frozen.before.json
$PY tests/test_frozen.py --update 2>&1 | tail -3
$PY tests/test_frozen.py 2>&1 | tail -3
echo "--- keys whose hash changed:"
$PY - <<'EOF'
import json
a, b = json.load(open("temp/t22/frozen.before.json")), json.load(open("frozen.json"))
a, b = a.get("files", a), b.get("files", b)
print(sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k)))
EOF
[ "$1" = dry ] && exit 0
cd /mnt/c/Auto-research
python3 autoresearch/tools/check_git.py --paths bev_fusion/FROZEN.md bev_fusion/frozen.json 2>&1 | tail -5
python3 autoresearch/tools/commit_paths.py -F bev_fusion/temp/t22/commit_msg_B.txt --trailer "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>" bev_fusion/FROZEN.md bev_fusion/frozen.json 2>&1 | tail -8
