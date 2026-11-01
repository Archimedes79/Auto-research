"""scratch: resume the stopped run t10.dual_long_s0 (ckpt_00004773.pt) and its chained eval through run.py, from the records.

main run: the SAME code as attempt 1 (commit d14597e2e8 passed to run_ar.sh as its second argument, run.py with --no-git so that it does not
snapshot the working tree: training/ carries other chats' uncommitted edits); the eval launcher is started again as it was (it is `blocked`
because its dependency ended with exit 75).
Run from bev_fusion/ in WSL:  python3 temp/autogt_session_scratch_2026-11-01/resume_t10.py [--dry]
"""
import json
import os
import subprocess
import sys

dry = "--dry" in sys.argv
env = dict(os.environ, AR_PROJECT="../bev_fusion")


def argv(run_id, extra_cmd=(), extra_flags=()):
    rec = json.load(open(f"records/runs/{run_id}/run.json", encoding="utf-8"))
    a = ["python3", "../autoresearch/tools/run.py", "--id", run_id, "--gpu", "--gpu-max-mb", "3000", "--split", rec["split"]]
    if rec.get("after"):
        a += ["--after", rec["after"]]
    a += ["--hyp", rec["hypothesis"], "--pred", rec["prediction"], "--rule", rec["rule"], *extra_flags, "--"]
    return a + rec["command"].split() + list(extra_cmd)


jobs = [("t10.dual_long_s0", argv("t10.dual_long_s0", ["d14597e2e8"], ["--no-git"])), ("t10.dual_long_s0.eval", argv("t10.dual_long_s0.eval"))]
for run_id, a in jobs:
    log = f"runs/{run_id.replace('.', '_')}.resume_launcher.log"
    print(run_id, "->", log, "\n   ", " ".join(x if len(x) < 60 else x[:50] + "..." for x in a))
    if not dry:
        subprocess.Popen(a, cwd=os.getcwd(), env=env, stdout=open(log, "a"), stderr=subprocess.STDOUT, start_new_session=True)
print("(dry run: nothing started)" if dry else "launched")
