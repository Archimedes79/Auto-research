"""Build the two shared files for the jobrunner commit from the CURRENT HEAD text plus my own hunk (never the working copy: other chats have uncommitted edits there)."""
import os
import subprocess

ROOT = "/mnt/c/Auto-research"
OUT = f"{ROOT}/bev_fusion/temp/cleanup/blobs"
os.makedirs(OUT, exist_ok=True)


def head(path):
    return subprocess.run(["git", "-C", ROOT, "show", f"HEAD:{path}"], capture_output=True, text=True, check=True).stdout


# autoresearch/README.md: one row after the runsuite row of the folder table
text = head("autoresearch/README.md")
lines = text.split("\n")
i = next(k for k, l in enumerate(lines) if l.startswith("| `trainsuite/`"))
row = ("| `jobrunner/` | run a method of any suite as a job under run.py (2026-11-10, after D74): the job names the call (`Class.method`) and its arguments - the data - and the catalogue turns them into "
       "typed references that are checked and give the dependencies; `check` / `submit` / `start` / `run` / `list` / `status`, the state read from run.py's own record; run.py stays the queue "
       "(`python3 -m jobrunner methods`, `selftest`; `README.md`) |")
assert not any(l.startswith("| `jobrunner/`") for l in lines)
lines.insert(i + 1, row)
open(f"{OUT}/autoresearch_README.md", "w", encoding="utf-8", newline="").write("\n".join(lines))

# .gitignore: the job folders of the job runner, next to the other launcher state
text = head(".gitignore")
assert ".jobs/" not in text
new = text.replace(".gpu_slots/\n", ".gpu_slots/\n.jobs/\n", 1)
assert new != text
open(f"{OUT}/gitignore", "w", encoding="utf-8", newline="").write(new)
print("ok", len(text.split("\n")), "->", len(new.split("\n")))
