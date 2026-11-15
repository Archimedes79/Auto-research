"""The diagnostics of the one-buffered-frame temporal fusion (t21, D51), promoted from scratch: `evaluation/stream_drift.py`, `stream_timing.py`,
`arm_guards.py`; CPU, no GPU, about 30 s (the drift smoke reads nuScenes mini and its cached features; it is skipped when they are missing).

  A  arm_guards: on a throw-away root with two runs (one without a warm card) the table has the cold and the warm block, the stored numbers to four
     decimals, nan for a missing metric, no column for a run without a card, and no block when no run has its card.
  B  stream_drift (tiny random model, scene-0061, 6 frames, pair and recurrent): the exact comparison (stream against the batch reference) agrees to
     float round-off (< 1e-3 on the logits, < 1e-3 on the velocity) at every reported step, the cold comparison is reported and finite; the json is
     written into STREAM_OUT (not into the repository).
  C  stream_timing (tiny model, CPU, 3 calls): every path of the six has a positive finite median for dt 0 and for the three time offsets
     (the light path only dt 0), the json is written into STREAM_OUT.
  D  the checks fail on a planted fault: a drift row with an exact difference of 0.5 is refused by the same limit.
Run (from the bev_fusion folder): wsl ~/miniconda3/envs/gcar/bin/python tests/test_stream_tools.py"""
import json
import math
import os
import subprocess
import sys
import tempfile
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import arm_guards  # noqa: E402

PY = sys.executable
fails, t0 = [], time.time()


def check(ok, what):
    print(("  ok    " if ok else "  FAIL  ") + what, flush=True)
    if not ok:
        fails.append(what)


def run(args, out):
    env = dict(os.environ, STREAM_OUT=out, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    return subprocess.run([PY] + args, cwd=ROOT, env=env, capture_output=True, text=True)


def exact_ok(rows, limit=1e-3):
    """every exact comparison of a drift json is finite and below the limit (None = not defined at that step)"""
    ex = [r["exact"] for r in rows if r["exact"]]
    return bool(ex) and all(math.isfinite(d[k]) and d[k] < limit for d in ex for k in ("sem_max", "v_max"))


# A ---------------------------------------------------------------------------------------------------------------------------------------------
print("A  arm_guards")
with tempfile.TemporaryDirectory() as root:
    for name, tags in (("run_a", ("", "_warm")), ("run_b", ("",)), ("run_none", ())):
        for tag in tags:
            d = os.path.join(root, "runs", name, "eval")
            os.makedirs(d, exist_ok=True)
            m = {"vel/dt+0.00/centre/field/moving/epe": 1.0836 if name == "run_a" else 1.2345, "detf/dt+0.00/mean/all/mAP": 0.4117}
            json.dump({"metrics": m}, open(os.path.join(d, f"card{tag}.json"), "w"))
    text = "\n".join(arm_guards.table(["run_a", "run_b", "run_none"], root))
    check("card.json (VAL, 1000 frames" in text and "card_warm.json" in text, "the cold and the warm block are printed")
    check("1.0836" in text and "1.2345" in text and "0.4117" in text, "the stored numbers appear with four decimals")
    check("nan" in text, "a metric a run does not have is nan")
    check("run_none" not in text, "a run without a card has no column")
    warm = text.split("card_warm.json")[1]
    check("run_b" not in warm and "run_a" in warm, "the warm block holds only the run that has a warm card")
    check(arm_guards.table(["run_none"], root) == [], "no block when no run has a card")

# B ---------------------------------------------------------------------------------------------------------------------------------------------
print("B  stream_drift (CPU smoke, nuScenes mini)")
with tempfile.TemporaryDirectory() as out:                     # t22: a camera_student chain is refused up front (its one-frame batches drop the image the student reads); no data needed
    cfgp = os.path.join(out, "student.json")
    json.dump({"model": {"camera_features": "dual", "camera_student": "resnet18", "student_trainable": True}}, open(cfgp, "w"))
    r = run(["evaluation/stream_drift.py", "--config", cfgp, "--buffer", "pair"], out)
    check(r.returncode != 0 and "camera_student" in r.stderr and "Traceback" not in r.stderr, "a camera_student chain is refused with a message that names the reason (no traceback)")
mini = os.path.expanduser("~/data/nuscenes/v1.0-mini")
if not (os.path.isdir(mini) and os.path.isdir(os.path.join(ROOT, "data", "cache", "cam_feats_mini"))):
    print("  skipped: nuScenes mini or data/cache/cam_feats_mini is missing")
else:
    with tempfile.TemporaryDirectory() as out:
        for buf in ("pair", "recurrent"):
            r = run(["evaluation/stream_drift.py", "--buffer", buf, "--frames", "6", "--steps", "1,2,3,4,6"], out)
            js = os.path.join(out, f"drift_scene-0061_{buf}_tiny.json")
            check(r.returncode == 0 and os.path.exists(js), f"{buf}: the script ends and writes its json into STREAM_OUT" + ("" if r.returncode == 0 else "\n" + r.stderr[-600:]))
            if os.path.exists(js):
                rows = json.load(open(js))["rows"]
                check(exact_ok(rows), f"{buf}: stream = batch reference to float round-off at every reported step ({len(rows)} steps)")
                cold = [r_["cold"] for r_ in rows if r_["cold"]]
                check(bool(cold) and all(math.isfinite(d["sem_mean"]) for d in cold), f"{buf}: the cold comparison is reported and finite")
                if buf == "pair":
                    bad = json.loads(json.dumps(rows))
                    bad[-1]["exact"]["sem_max"] = 0.5
                    check(not exact_ok(bad), "D  a planted exact difference of 0.5 is refused by the same limit")

# C ---------------------------------------------------------------------------------------------------------------------------------------------
print("C  stream_timing (tiny model, CPU)")
with tempfile.TemporaryDirectory() as out:
    r = run(["evaluation/stream_timing.py", "--size", "tiny", "--device", "cpu", "--n", "3"], out)
    js = os.path.join(out, "timing_tiny_cpu.json")
    check(r.returncode == 0 and os.path.exists(js), "the script ends and writes its json into STREAM_OUT" + ("" if r.returncode == 0 else "\n" + r.stderr[-600:]))
    if os.path.exists(js):
        rows = json.load(open(js))["ms_per_call_median"]
        check(len(rows) == 6, f"six paths timed ({len(rows)})")
        vals = [v for per in rows.values() for v in per.values() if v is not None]
        check(len(vals) == 11 and all(math.isfinite(v) and v > 0 for v in vals), f"every timed median is positive and finite ({len(vals)} values; the light path has dt 0 only)")

print(f"\n{'FAILED: ' + str(len(fails)) + ' check(s)' if fails else 'test_stream_tools passed'} ({time.time() - t0:.0f} s)")
sys.exit(1 if fails else 0)
