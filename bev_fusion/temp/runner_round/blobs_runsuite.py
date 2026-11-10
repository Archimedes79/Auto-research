"""Scratch: blobs for the shared files of the runsuite commit = HEAD text + only my change.
  autoresearch/README.md: my row of runsuite/ and the three class words of the convention (Runner next to RunJobs)
  bev_fusion/tests/README.md: the row of test_runner_predict.py after the row of test_evaluation_api.py"""
import os
import subprocess

REPO = "/mnt/c/Auto-research"
OUT = os.path.join(REPO, "bev_fusion/temp/cleanup/blobs")
os.makedirs(OUT, exist_ok=True)


def head(path):
    return subprocess.run(["git", "-C", REPO, "show", f"HEAD:{path}"], capture_output=True, text=True, check=True).stdout


def write(path, lines):
    dest = os.path.join(OUT, path.replace("/", "_"))
    with open(dest, "w", encoding="utf-8", newline="") as f:
        f.write("\n".join(lines))
    new, old = "\n".join(lines).split("\n"), head(path).split("\n")
    changed = sum(1 for a, b in zip(new, old) if a != b) + abs(len(new) - len(old))
    print(f"{path}: {changed} line(s) differ from HEAD -> {dest}")


# 1 autoresearch/README.md
row = ("| `runsuite/` | run a trained network, in three layers: the command line (`cli.py`), the class `Runner` (`api.py`: `predict` / `plan` / `list_results` / `tap` / `time` / `probe`, the network on frames "
       "without ground truth (D72), always warm, stored as a ResultSet that `Evaluation.score` reads; no queue) and the functionality (`engine.py`); beside them the class `RunJobs` (`jobs.py`, the interface of the "
       "training: `check` / `run` / `status` / `queue_add`; a runner job is an item of kind `run` in the same queue as the trainings; the command comes from the `runsuite` block of `project.json`); "
       "`python -m runsuite`, selftest `python -m runsuite.selftest` |")
text = head("autoresearch/README.md")
lines = text.split("\n")
i = next(k for k, l in enumerate(lines) if l.startswith("| `runsuite/` |"))
lines[i] = row
text = "\n".join(lines)
for old, new in (("`runsuite` (class `RunJobs`, the jobs of the runner of a trained network)", "`runsuite` (class `Runner`, a trained network on frames, and class `RunJobs`, the jobs of the project's runner)"),
                 ("`Training(project)`, `RunJobs(project)`;", "`Training(project)`, `Runner(project)`, `RunJobs(project)`;"),
                 ("(`TrainingError`, `RunJobsError` = the same class)", "(`TrainingError` = `RunJobsError`; `RunnerError` = the evaluation's `EvaluationError`)")):
    assert text.count(old) == 1, old
    text = text.replace(old, new)
write("autoresearch/README.md", text.split("\n"))

# 2 bev_fusion/tests/README.md
rows = ["| `test_runner_predict.py` | the tool end to end on a REAL run (default `runs/t20_base_s0`), CPU, 3 frames of val scene-0003, about 10 minutes, not in `run_gates.sh`, through the command lines of the suites on a copy of the run folder: `Runner.plan`, `Runner.predict --taps fused` (every dt, detections, no loss, no ground-truth signature, the memory fill 0.00 at the first frame and growing), `Runner.list_results`, `Runner.tap`, the example job file `records/jobs/running/pred_t20_small.json` passes `RunJobs.check`, and the predictions are scored later by `Evaluation.score` (finite numbers, no loss among them) |"]
lines = head("bev_fusion/tests/README.md").split("\n")
i = next(k for k, l in enumerate(lines) if l.startswith("| `test_evaluation_api.py` |"))
write("bev_fusion/tests/README.md", lines[:i + 1] + rows + lines[i + 1:])
