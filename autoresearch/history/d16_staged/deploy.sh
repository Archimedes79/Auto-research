#!/bin/bash
# D16 deploy by copy. Run by Mirko in the WSL (the gate hook protects autoresearch/tools/ and METHOD.md, so the agent does not write there):
#   wsl bash autoresearch/history/d16_staged/deploy.sh
set -e
D="$(cd "$(dirname "$0")" && pwd)"
A="$(cd "$D/../.." && pwd)"
for f in run tick schedule ledger selftest; do cp "$D/$f.py.staged" "$A/tools/$f.py"; done
cp "$D/METHOD.patched.md" "$A/METHOD.md"
cd "$A/.." && git status --short autoresearch/tools autoresearch/METHOD.md
echo "deployed; next: wsl python3 autoresearch/tools/selftest.py"
