"""Scratch: blob of bev_fusion/tests/README.md = HEAD text + only my three rows changed (the working copy is a stale copy of another chat: not used)"""
import os
import subprocess

REPO = "/mnt/c/Auto-research"
path = "bev_fusion/tests/README.md"
head = subprocess.run(["git", "-C", REPO, "show", f"HEAD:{path}"], capture_output=True, check=True).stdout.decode("utf-8")
new_rows = {
    "| `test_runner_class.py` |": "| `test_runner_class.py` | the class `Runner` (`runner/runner.py`) and its command line `python -m runner` on the toy network of `autoresearch/evalsuite/selftest_api.py`: importing the Runner loads neither torch, nor trainsuite, nor the command-line generator, the six methods are the six commands (summaries, no other public method), `plan` finds problems and runs nothing, `predict` stores without reading a ground truth and a second `predict` reuses, `tap` / `time` / `probe`, the command line (`--json`, exit 0 / 1 / 2, `methods`); no data, seconds; in `run_gates.sh` |",
    "| `test_runner_real.py` |": "| `test_runner_real.py` | the `Runner` end to end on a REAL run (default `runs/t20_base_s0`, argument: another run with the restructured code; checked on `t16_full_s0`, `t20_base_s0`, `t23_C_s1`), CPU, 3 frames of val scene-0003, about 10 minutes and 6 GB, not in `run_gates.sh`, through `python -m runner` on a copy of the run folder: `Runner.plan`, `Runner.predict --taps fused` (every dt, detections, no loss, no ground-truth signature, the memory fill 0.00 at the first frame and growing; the ids of plan and predict are compared unless the code that makes a prediction was edited meanwhile), `Runner.list_results`, `Runner.tap`, and the predictions scored later by `Evaluation.score` (finite numbers, no loss among them) |",
    "| `test_suites_conform.py` |": "| `test_suites_conform.py` | `Training`, `Evaluation` and `Runner` against the one convention of `autoresearch/suitecli.py`: `methods` with descriptions, `--json` is one document (the `Runner.list_results` probe is skipped with a message when `runs/` holds no run), exit 0 / 1 / 2 (also for an unknown id), verbs first, layering (the evaluation's core imports no torch and not the runner, `runner.py` only numpy / evalsuite, `__main__.py` only the runner and suitecli); about 40 s; in `run_gates.sh` |",
}
lines = head.split("\n")
done = 0
for k, row in new_rows.items():
    i = next(n for n, l in enumerate(lines) if l.startswith(k))
    lines[i] = row
    done += 1
dest = os.path.join(REPO, "bev_fusion/temp/blob_round_tests_README.md")
open(dest, "w", encoding="utf-8", newline="").write("\n".join(lines))
print(f"{done} rows replaced -> {dest}; differs from HEAD in {sum(1 for a, b in zip(lines, head.split(chr(10))) if a != b)} lines")
