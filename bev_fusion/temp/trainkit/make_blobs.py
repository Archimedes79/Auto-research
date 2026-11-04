"""scratch: my version of autoresearch/README.md = HEAD + only my trainkit row (for commit_paths.py --blob); replaces the row if HEAD has it"""
import subprocess
import sys

out = sys.argv[1]
row = open(sys.argv[2], encoding="utf-8").read().rstrip("\n")
readme = subprocess.run(["git", "show", "HEAD:autoresearch/README.md"], capture_output=True, check=True).stdout.decode("utf-8")
lines = readme.split("\n")
have = [k for k, ln in enumerate(lines) if ln.startswith("| `trainkit/` |")]
if have:
    lines[have[0]] = row
else:
    lines.insert(next(k for k, ln in enumerate(lines) if ln.startswith("| `trainsuite/` |")) + 1, row)
open(out + "/README_autoresearch.md", "w", encoding="utf-8", newline="").write("\n".join(lines))
print("blob written to", out)
