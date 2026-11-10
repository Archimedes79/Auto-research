"""Scratch: the blob of bev_fusion/tests/README.md for the runner commit = HEAD text + only my three rows (after the row of test_evaluation_api.py)"""
import os
import subprocess

REPO = "/mnt/c/Auto-research"
OUT = os.path.join(REPO, "bev_fusion/temp/cleanup/blobs")
os.makedirs(OUT, exist_ok=True)
path = "bev_fusion/tests/README.md"
head = subprocess.run(["git", "-C", REPO, "show", f"HEAD:{path}"], capture_output=True, check=True).stdout.decode("utf-8")
rows = [
    "| `test_runner_class.py` | the class `Runner` (`runner/runner.py`) and its command line `python -m runner` on the toy network of `autoresearch/evalsuite/selftest_api.py`: importing the Runner loads neither torch nor a suite besides evalsuite, the six methods are the six commands (summaries, helpers are no commands), `plan` finds problems and runs nothing, `predict` stores without reading a ground truth and a second `predict` reuses, `tap` / `time` / `probe`, the command line (`--json`, exit 0 / 1 / 2, `methods`); no data, seconds; in `run_gates.sh` |",
    "| `test_runner_real.py` | the `Runner` end to end on a REAL run (default `runs/t20_base_s0`), CPU, 3 frames of val scene-0003, about 10 minutes and 6 GB, not in `run_gates.sh`, through `python -m runner` on a copy of the run folder: `Runner.plan`, `Runner.predict --taps fused` (every dt, detections, no loss, no ground-truth signature, the memory fill 0.00 at the first frame and growing), `Runner.list_results`, `Runner.tap`, and the predictions scored later by `Evaluation.score` (finite numbers, no loss among them) |",
    "| `test_suites_conform.py` | `Training`, `Evaluation` and `Runner` against the one convention of `autoresearch/suitecli.py`: `methods` with descriptions, `--json` is one document, exit 0 / 1 / 2 (also for an unknown id), verbs first, layering (the evaluation's core imports no torch and not the runner, `runner.py` only numpy / suitecli / evalsuite, `__main__.py` only the runner and suitecli); about 40 s; in `run_gates.sh` |",
]
lines = head.split("\n")
i = next(k for k, l in enumerate(lines) if l.startswith("| `test_evaluation_api.py` |"))
new = "\n".join(lines[:i + 1] + rows + lines[i + 1:])
dest = os.path.join(OUT, path.replace("/", "_"))
open(dest, "w", encoding="utf-8", newline="").write(new)
print(f"{path}: {len(new.split(chr(10))) - len(lines)} lines more than HEAD -> {dest}")
