import subprocess

head = subprocess.run(["git", "show", "HEAD:bev_fusion/tests/README.md"], capture_output=True, text=True, encoding="utf-8", check=True).stdout
old = ("the class interface `Evaluation` on the BEV system (`autoresearch/evalsuite/api.py`, adapter `evaluation/bev_eval.py`): without option (CPU, seconds) the interface's own checks on a toy project "
       "(`python -m evalsuite.selftest_api`: frames, result set and writer lock, run = the engine's run on the kept frames, reuse of stored predictions also for a subset, rescoring under another GT "
       "without the network and the stale-loss warning, a run killed half-way resumes with the right memory, predict + score = run, `check` finds problems first, the command line: clean json, exit codes, "
       "no collision with the older words), the adapter's identity (a retrained checkpoint of the same name is another one), the run resolution, TEST and unknown metrics refused; "
       "`--real [RUN]` (CPU, 4 frames, about 5 minutes): ")
new = ("the class interface `Evaluation` + `Runner` on the BEV system (`autoresearch/evalsuite/api.py`, `runner_api.py`, adapter `evaluation/bev_eval.py`): without option (CPU, seconds) the interface's own "
       "checks on a toy project (`python -m evalsuite.selftest_api`: frames, result set with its writer lock (a second process), damaged / foreign / emptied frame folders, run = the engine's run on the kept "
       "frames, reuse of stored predictions also for a subset, rescoring under another GT without the network, a run killed half-way resumes with the right memory, Runner.run + score = run, the identity "
       "changing while the network loads, the health of the memory at its boundaries, `check` finds problems first, the command line: clean json, exit codes, `Runner.` words), the adapter's identity "
       "(a retrained checkpoint of the same name, an edit of bev_free.py is another one; the Runner's pair free_source / free_subject), the run resolution, TEST and unknown metrics refused; "
       "`--real [RUN]` (default `runs/t20_base_s0`, CPU, 4 frames, about 5 minutes, skips below 18 GB of free RAM): ")
assert head.count(old) == 1, head.count(old)
open("bev_fusion/temp/pitch/readme_tests_mine.md", "w", encoding="utf-8", newline="").write(head.replace(old, new))
print("ok")
