"""blob of a shared file = HEAD + only MY hunks (scratch). python mkblob.py FILE OUT [REGEX of the other chat's hunks, default 'GtGenerator|autogt']
A hunk of `git diff -U0` whose changed lines match the regex is left out (it belongs to the other chat)."""
import re
import subprocess
import sys

path, out = sys.argv[1], sys.argv[2]
skip = re.compile(sys.argv[3] if len(sys.argv) > 3 else "GtGenerator|autogt")
head = subprocess.run(["git", "show", f"HEAD:{path}"], capture_output=True, check=True).stdout.decode("utf-8").split("\n")
diff = subprocess.run(["git", "diff", "-U0", "--", path], capture_output=True, check=True).stdout.decode("utf-8")
hunks, cur = [], None
for line in diff.split("\n"):
    m = re.match(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", line)
    if m:
        cur = {"a": int(m.group(1)), "n": 1 if m.group(2) is None else int(m.group(2)), "minus": [], "plus": []}
        hunks.append(cur)
    elif cur is not None and line.startswith("-") and not line.startswith("---"):
        cur["minus"].append(line[1:])
    elif cur is not None and line.startswith("+") and not line.startswith("+++"):
        cur["plus"].append(line[1:])
kept = [h for h in hunks if not skip.search("\n".join(h["minus"] + h["plus"]))]
print(f"{path}: {len(hunks)} hunks, {len(hunks) - len(kept)} left to the other chat, {len(kept)} mine")
lines = list(head)
for h in sorted(kept, key=lambda h: -h["a"]):
    start = h["a"] if h["n"] > 0 else h["a"] + 0          # for n == 0 (pure insertion) 'a' is the line AFTER which to insert
    if h["n"] == 0:
        lines[h["a"]:h["a"]] = h["plus"]
    else:
        assert lines[h["a"] - 1:h["a"] - 1 + h["n"]] == h["minus"], (path, h["a"])
        lines[h["a"] - 1:h["a"] - 1 + h["n"]] = h["plus"]
open(out, "w", encoding="utf-8", newline="\n").write("\n".join(lines))
