"""Scratch: take my runsuite hunks out of the working copy of autoresearch/README.md (other chats' hunks stay): the row of runsuite/ back to HEAD's, the four class words back to RunJobs"""
import subprocess

REPO = "/mnt/c/Auto-research"
path = REPO + "/autoresearch/README.md"
head = subprocess.run(["git", "-C", REPO, "show", "HEAD:autoresearch/README.md"], capture_output=True, check=True).stdout.decode("utf-8")
cur = open(path, encoding="utf-8", newline="").read()
eol = "\r\n" if "\r\n" in cur else "\n"
head_row = next(l for l in head.splitlines() if l.startswith("| `runsuite/` |"))
lines = cur.split(eol)
i = next(k for k, l in enumerate(lines) if l.startswith("| `runsuite/` |"))
lines[i] = head_row
cur = eol.join(lines)
for new, old in (("`runsuite` (class `Runner`, the jobs of the runner of a trained network)", "`runsuite` (class `RunJobs`, the jobs of the runner of a trained network)"),
                 ("`Training(project)`, `Runner(project)`;", "`Training(project)`, `RunJobs(project)`;"),
                 ("(`TrainingError`, `RunnerError` = the same class)", "(`TrainingError`, `RunJobsError` = the same class)"),
                 ("belongs to `Training` and `Runner` only: the other suites have no queue.", "belongs to `Training` and `RunJobs` only: the other suites have no queue.")):
    assert cur.count(new) == 1, new
    cur = cur.replace(new, old)
open(path, "w", encoding="utf-8", newline="").write(cur)
print("reverted; remaining mentions of Runner:", [l[:90] for l in cur.split(eol) if "`Runner`" in l])
