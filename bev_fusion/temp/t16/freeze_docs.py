import subprocess
def head(p):
    return subprocess.run(["git", "show", f"HEAD:{p}"], capture_output=True, text=True, check=True, encoding="utf-8").stdout
def once(s, a, b):
    assert s.count(a) == 1, a[:60]
    return s.replace(a, b)
# run_gates.sh (the working copy equals HEAD)
s = head("bev_fusion/tests/run_gates.sh")
a = 'run "clip split"'
line = [l for l in s.split("\n") if l.startswith(a)][0]
s = s.replace(line, line + '\nrun "frozen frame"      "$PY" tests/test_frozen.py                 # the frame of FROZEN.md has the recorded hashes (a bug fix updates them with a changelog line)')
open("bev_fusion/tests/run_gates.sh", "w", encoding="utf-8", newline="\n").write(s)
# tests/README.md as a blob (the working copy is stale)
t = head("bev_fusion/tests/README.md").split("\n")
i = [k for k, l in enumerate(t) if l.startswith("| `test_clip_split.py` |")][0]
t.insert(i + 1, "| `test_frozen.py` | the frozen frame (`../FROZEN.md`, `../frozen.json`): the listed files of `training/`, `evaluation/` and `runner/` have the recorded SHA-256 (line endings ignored); `--update` rewrites the hashes after a bug fix, `--selftest` on a toy folder (clean copy passes, an edited and a missing file fail) | an edit of a frozen file fails with its name | passes, 24 files |")
open("bev_fusion/temp/t16/tests_README_frozen.md", "w", encoding="utf-8").write("\n".join(t))
# bev_fusion/README.md (the working copy equals HEAD): a status bullet
r = head("bev_fusion/README.md")
a = "- **Open:** `records/open_findings_2026-11-02.md`"
r = once(r, a, "- **Frozen:** the frame (trainer, data path, interfaces, evaluation, runner) is fixed since 2026-11-02, the work is in the modules: `FROZEN.md` lists both sides and the rule for a bug fix, `tests/test_frozen.py` checks the hashes (`frozen.json`), tag `bf/frame-2026-11-02`.\n" + a)
open("bev_fusion/README.md", "w", encoding="utf-8", newline="\n").write(r)
print("ok")
