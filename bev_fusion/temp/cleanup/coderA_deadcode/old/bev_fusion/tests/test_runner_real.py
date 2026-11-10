"""The Runner end to end on a REAL run (default runs/t20_base_s0), CPU, the first 3 frames of val scene-0003, about 10 minutes and 6 GB (nuScenes, the checkpoint), not in run_gates.sh. Everything
goes through the command lines in subprocesses (one network per process), on a copy of the run folder (its config, code snapshot, the checkpoint linked), so no real run folder is written:
  1  `python -m runner Runner.plan`: ok, 3 frames to run, none stored
  2  `Runner.predict --taps fused`: 3 frames predicted, no warning, the id <network>@r<hash>; the stored frames hold every dt, the detections, the fused tap, the memory fill 0.00 at the first frame and
     growing, the device, and NO loss and no ground-truth signature (D72)
  3  `Runner.list_results` lists it, `Runner.tap` gives the shape of the fused tap
  4  the predictions are scored LATER: `Evaluation.score <id> <frames>` gives finite headline numbers (the run reads no ground truth, the scoring does)
Run (from the bev_fusion folder): ~/miniconda3/envs/gcar/bin/python tests/test_runner_real.py [runs/<name>]"""
import glob
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
AR = os.path.abspath(os.path.join(ROOT, "..", "autoresearch"))
sys.path[:0] = [AR, ROOT]
RUN = sys.argv[1] if len(sys.argv) > 1 else "runs/t20_base_s0"
SCENE, N = "scene-0003", 3
FRAMES = "frames:" + ",".join(f"{SCENE}/{k}" for k in range(N)) + ":val"
ENV = {**os.environ, "OMP_NUM_THREADS": "3", "MKL_NUM_THREADS": "3"}


def call(*argv):
    """`python -m runner ...` from the bev_fusion folder -> (exit code, stdout, stderr)"""
    p = subprocess.run([sys.executable, "-m", "runner", *argv], cwd=ROOT, env=ENV, capture_output=True, text=True, timeout=3000)
    return p.returncode, p.stdout, p.stderr


def api(code):
    """run python code in a process of its own with Runner and Evaluation imported; its last print is a json document -> the document. An id carries only the name of a network, so a call that
    reads stored results of the COPY of the run folder first tells the object where the network is (`resolve`): the real folder of the same name is not the one that holds them"""
    p = subprocess.run([sys.executable, "-c", f"import json, sys\nsys.path[:0] = [{ROOT!r}]\nfrom runner.runner import Runner\nfrom evalsuite.api import Evaluation\n{code}"], cwd=ROOT, env=ENV,
                       capture_output=True, text=True, timeout=3000)
    assert p.returncode == 0, p.stderr[-600:]
    return json.loads(p.stdout.strip().splitlines()[-1])


def main():
    if not glob.glob(os.path.join(ROOT, RUN, "ckpt_*.pt")):
        print(f"SKIP: {RUN} has no checkpoint here")
        return
    src_run = os.path.join(ROOT, RUN)
    os.makedirs(os.path.join(ROOT, "temp"), exist_ok=True)
    with tempfile.TemporaryDirectory(dir=os.path.join(ROOT, "temp")) as tmp:
        net = os.path.relpath(os.path.join(tmp, os.path.basename(RUN)), ROOT)
        os.makedirs(os.path.join(ROOT, net))
        for f in ("config.json", "gt.json"):
            if os.path.isfile(os.path.join(src_run, f)):
                shutil.copy(os.path.join(src_run, f), os.path.join(ROOT, net))
        shutil.copytree(os.path.join(src_run, "code"), os.path.join(ROOT, net, "code"))
        ck = sorted(glob.glob(os.path.join(src_run, "ckpt_*.pt")))[-1]
        os.symlink(ck, os.path.join(ROOT, net, os.path.basename(ck)))

        # 1 the dry run
        rc, out, err = call("Runner.plan", net, FRAMES, "--taps", "fused", "--workers", "0", "--json")
        assert rc == 0, (rc, err[-400:])
        plan = json.loads(out)
        assert plan["ok"] and plan["frames_kept"] == N and plan["frames_stored"] == 0 and plan["frames_to_run"] == N, plan
        print(f"  1 ok: plan: {plan['frames_to_run']} frames to run, nothing stored, id {plan['results']}")

        # 2 the run without ground truth
        rc, out, err = call("Runner.predict", net, FRAMES, "--taps", "fused", "--device", "cpu", "--workers", "0", "--progress", "0", "--json")
        assert rc == 0, (rc, err[-600:])
        res = json.loads(out)
        assert res["results"] == plan["results"] and res["frames_predicted"] == N and res["predictions_reused"] is False and res["warnings"] == [], res
        from evalsuite.results import ResultSet
        rid = res["results"].partition("@")[2]
        rs = ResultSet.open(os.path.join(ROOT, net, "eval", "results", rid))
        refs = rs.refs()
        assert len(refs) == N and rs.n_frames() == N and "gt_signature" not in rs.manifest["identity"]["inputs"], (len(refs), rs.manifest["identity"]["inputs"])
        fill, loaded = [], [rs.load(r) for r in refs]
        for p in loaded:
            assert sorted(p.outputs["dt"]) == [-0.5, 0.0, 0.5] and "loss" not in p.info and not any(str(k).startswith("group.") for k in p.info) and p.info.get("device") == "cpu", p.info
            o = p.outputs["dt"][0.0]
            assert o["prob"] and all(np.isfinite(a).all() for a in o["prob"].values()) and o["dets"], "maps and detections of dt 0"
            fill.append(p.info["memory_fill"])
        assert fill[0] <= 0.05 and fill[-1] > fill[0], f"the memory must be empty at the first frame and grow: {fill}"
        assert rs.tap_names(refs[0]) == ["fused"], rs.tap_names(refs[0])
        print(f"  2 ok: predicted {N} frames without ground truth: every dt, detections, no loss, device cpu, memory fill {[round(f, 3) for f in fill]}, taps {rs.tap_names(refs[0])}, {rs.size_mb()} MB")

        # 3 reading it back
        rc, out, err = call("Runner.list_results", net, "--json")
        assert rc == 0 and [x["id"] for x in json.loads(out)] == [res["results"]], (rc, out, err[-300:])
        tap = api(f"r = Runner(); r.resolve({net!r}); print(json.dumps(r.tap({res['results']!r}, '{SCENE}/1', 'fused')))")
        assert tap["tap"] == "fused" and len(tap["shape"]) == 3 and tap["dtype"].startswith("float") and os.path.isfile(tap["path"]), tap
        print(f"  3 ok: list_results, tap {tap['shape']} {tap['dtype']}")

        # 4 scored later against the ground truth
        nums = api(f"ev = Evaluation(); ev.resolve({net!r}); s = ev.score({res['results']!r}, {FRAMES!r}, workers=0, progress=0); "
                   f"print(json.dumps(ev.show_numbers(s['id'])['numbers'], default=float))")
        finite = {k: v for k, v in nums.items() if isinstance(v, (int, float)) and math.isfinite(v)}
        assert len(finite) >= 10 and not any(k.startswith(("loss.", "val_loss/")) for k in nums), (len(finite), [k for k in nums if "loss" in k][:3])
        print(f"  4 ok: scored later: {len(finite)} finite numbers, no loss among them (e.g. {sorted(finite)[0]})")
    print(f"test_runner_real passed: Runner.plan / predict / list_results / tap and Evaluation.score on {N} frames of {SCENE} through the command lines, run {os.path.basename(RUN)}")


if __name__ == "__main__":
    main()
