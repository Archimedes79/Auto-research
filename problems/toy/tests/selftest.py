"""The toy problem end to end with the real tools, in a throw-away copy (nothing of this folder is written):
known answers of data, truth and metric; two arms through Training and run.py to cards, a paired comparison and a verdict of decide.py; the refusals of the way;
a training stopped in the middle and resumed equals the uninterrupted one.

    bash tools/py.sh tests/selftest.py [--quick]      (from the problem folder, under WSL; --quick: the known answers only; about 1 - 2 minutes in full)
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
import torch

from data import make_data
from evaluation import adapter as AD
from Evaluation import FrameRef, Prediction, Sample
from gt import generate as GT
from gt import raster
from net.prep import shards as S

ROOT = S.ROOT
AR = os.environ.get("AR_ROOT") or os.path.join(os.path.dirname(os.path.dirname(ROOT)), "autoresearch")


def known_answers():
    # truth
    one = raster.rasterise([(16.5, 16.5, 0.5)], 32)[0]
    assert one.sum() == 1 and one[16, 16], one.sum()
    assert raster.rasterise([(16.0, 16.0, 100.0)], 32).all() and not raster.rasterise([(16.0, 16.0, 0.0)], 32).any()
    assert GT.signature(4) != GT.signature(2) and GT.signature(4) == GT.signature(4)
    # data: the recipe makes the bytes the manifest names, and a scene does not depend on the others
    stored = {s["name"]: s["sha256"] for s in S.manifest()["shards"]}
    built = {e["name"]: e["sha256"] for e, _ in make_data.build(S.manifest()["recipe"])}
    assert built == stored, "this machine makes other data than the manifest says"
    a, b = make_data.scene(make_data.RECIPE, 0, 3), make_data.scene(make_data.RECIPE, 0, 3)
    assert np.array_equal(a[1], b[1]) and not np.array_equal(a[1], make_data.scene(make_data.RECIPE, 1, 3)[1])
    # metric
    metric, truth = AD.SegIoU(), np.zeros((4, 4), bool)
    truth[0, :2] = True
    sample = Sample(FrameRef("s", 0), inputs=None, reference=truth)
    value = lambda pred: metric.reduce([metric.frame(sample, Prediction(outputs=pred))])["iou"]        # noqa: E731
    shifted = np.roll(truth, 1, axis=1)
    assert value(truth) == 1.0 and value(~truth) == 0.0 and abs(value(shifted) - 1 / 3) < 1e-12, value(shifted)
    assert metric.support(metric.frame(sample, Prediction(outputs=truth))) == {"iou": 2.0}
    print("ok  known answers: truth, data bytes, metric")


def sh(cmd, cwd, env=None, ok=(0,)):
    r = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)
    assert r.returncode in ok, f"{' '.join(cmd)} -> {r.returncode}\n{r.stdout[-1500:]}\n{r.stderr[-1500:]}"
    return r


def copy_of_the_problem(tmp):
    proj = os.path.join(tmp, "toy")
    skip = shutil.ignore_patterns("runs", "store", "records", "results", "temp", "reports", "experiments", "__pycache__", "*.lock")
    shutil.copytree(ROOT, proj, ignore=skip)
    for cmd in (["git", "init", "-q"], ["git", "config", "user.name", "toy selftest"], ["git", "config", "user.email", "selftest@example.invalid"], ["git", "add", "-A"],
                ["git", "commit", "-qm", "copy"]):
        sh(cmd, proj)
    return proj


def write_card(proj):
    folder = os.path.join(proj, "experiments", "E9_chain")
    os.makedirs(folder)
    card = {"id": "E9", "title": "chain test", "kind": "variant", "metric": "iou", "direction": "higher", "band": 0.01, "prediction": {"value": 0.0, "vs": "c"},
            "arms": {k: {"recipe": f"width={w}", "names": [f"E9_{k}_s0"], "run_ids": [f"E9.{k}_s0"], "seeds": [0]} for k, w in (("c", 4), ("a", 6))},
            "control": "c", "rule": [{"metric": "iou", "min_effect": 0.01}], "evaluate": {"split": "val", "compare": "c"}}
    with open(os.path.join(folder, "experiment.md"), "w", encoding="utf-8") as f:
        f.write("# E9 chain test\n\n```json card\n" + json.dumps(card, indent=1) + "\n```\n")


def wait(proj, run_id, seconds=180):
    path = os.path.join(proj, "records", "runs", run_id, "run.json")
    for _ in range(seconds * 2):
        try:
            with open(path, encoding="utf-8") as f:
                status = json.load(f)["status"]
        except (OSError, ValueError, KeyError):
            status = "none"
        if status in ("ok", "failed", "stopped", "timeout", "blocked", "gpu_busy"):
            return status
        time.sleep(0.5)
    raise AssertionError(f"{run_id} did not end in {seconds} s")


def chain(proj, env):
    from Training import Training, TrainingError
    sh(["bash", "tools/prepare.sh", "have", "training/configs/base.json"], proj, env, ok=(1,))      # nothing there yet: "missing", not an error
    sh(["bash", "tools/prepare.sh", "make", "training/configs/base.json"], proj, env)
    sh(["bash", "tools/py.sh", "gt/generate.py", "check"], proj, env)
    write_card(proj)
    os.environ.update(env)
    training = Training(proj)
    base = {"config": "training/configs/base.json", "preset": ["smoke"]}
    try:
        training.run({**base, "id": "E9.x_s0", "set": {"seed": 0}})
    except TrainingError as e:
        assert "card" in str(e), e                                                                  # a run without a card and without no_card is refused
    else:
        raise AssertionError("a run without a card started")
    bad = training.check({**base, "id": "E9.x_s0", "set": {"widht": 3}, "no_card": True})
    assert not bad["ok"] and "widht" in json.dumps(bad), bad                                        # a typo in a setting is caught by the static check
    for letter, width, extra in (("c", 4, {}), ("a", 6, {"after": ["E9.c_s0"]})):
        settings = {"width": width, "seed": 0, **({"compare_to": "E9_c_s0"} if letter == "a" else {})}
        training.run({**base, "id": f"E9.{letter}_s0", "set": settings, "card": {"hyp": "chain", "pred": "none", "rule": "iou +0.01"}, **extra})
    for run_id in ("E9.c_s0", "E9.a_s0"):
        status = wait(proj, run_id)
        log = os.path.join(proj, "records", "runs", run_id, "run.log")
        assert status == "ok", f"{run_id} {status}\n" + (open(log, encoding="utf-8").read()[-2000:] if os.path.exists(log) else "")
    cards = {k: json.load(open(os.path.join(proj, "results", "cards", f"E9_{k}_s0", "val.json"), encoding="utf-8")) for k in ("c", "a")}
    assert cards["a"]["compare"]["name"] == "E9_c_s0" and {"diff", "lo", "hi"} <= set(cards["a"]["compare"]["paired"]["iou"])
    assert cards["a"]["provenance"]["evaluator"]["dirty"] == [] and cards["a"]["frames"] == 160, cards["a"]["provenance"]
    ar_env = {**env, "AR_PROJECT": proj}
    sh(["python3", os.path.join(AR, "tools", "card.py"), "check", "--all"], proj, ar_env)
    out = sh(["python3", os.path.join(AR, "tools", "decide.py"), "--experiment", "E9", "--write"], proj, ar_env).stdout
    verdict = next(v for v in ("KEEP", "CUT", "TIE", "INCONCLUSIVE") if f"E9: {v}" in out)
    ledger = open(os.path.join(proj, "records", "LEDGER.csv"), encoding="utf-8").read()
    assert f"decide.py" in ledger and "E9.a_s0" in ledger
    assert training.status("E9.a_s0")["state"] == "done" and os.path.isfile(training.plot("E9.a_s0")["png"])
    assert sh(["git", "tag", "-l", "toy/E9.*"], proj).stdout.split() == ["toy/E9.a_s0", "toy/E9.c_s0"]
    record = os.path.join(proj, "results", "cards", "E9_a_s0", "val.json")
    kept = open(record, "rb").read()
    by_hand = {k: v for k, v in env.items() if k not in ("AR_COMMIT", "AR_RUN_ID")}
    sh(["bash", "tools/py.sh", "evaluation/score.py", os.path.join(proj, "runs", "E9_a_s0", "config.json")], proj, by_hand)
    assert open(record, "rb").read() == kept, "a call by hand replaced a record"
    assert "compare" in json.load(open(os.path.join(proj, "temp", "cards", "E9_a_s0", "val.json"), encoding="utf-8"))
    print(f"ok  chain: two arms through Training and run.py, cards with the paired block, verdict {verdict}, status, curves, tags; refusals: no card, a typo; "
          "a call by hand leaves the records alone")


def stop_and_resume(proj, env):
    from trainkit import control
    base = json.load(open(os.path.join(proj, "training", "configs", "base.json"), encoding="utf-8"))
    base.update(steps=3000, save_every=500, eval_every=3000, log_every=1, threads=1)
    os.makedirs(os.path.join(proj, "temp"), exist_ok=True)

    def start(name):
        cfg = os.path.join(proj, "temp", name + ".json")
        json.dump({**base, "name": name}, open(cfg, "w", encoding="utf-8"))
        return ["bash", "tools/py.sh", "training/train.py", cfg]

    sh(start("whole"), proj, env)
    cmd = start("cut")
    process = subprocess.Popen(cmd, cwd=proj, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    log = os.path.join(proj, "runs", "cut", "log.jsonl")
    while process.poll() is None and not (os.path.exists(log) and os.path.getsize(log) > 0):
        time.sleep(0.02)
    control.request_stop(os.path.join(proj, "runs", "cut"), "selftest")
    out = process.communicate(timeout=180)[0]
    assert process.returncode == control.EXIT_STOPPED, f"exit {process.returncode} (a run that ended before the stop request proves nothing)\n{out[-800:]}"
    sh(cmd, proj, env)

    def final(name):
        ck = torch.load(os.path.join(proj, "runs", name, "ckpt_00003000.pt"), map_location="cpu", weights_only=False)
        return {**{k: v for k, v in ck["state"].items()}, **{"ema." + k: v for k, v in ck["ema"].items()}}

    a, b = final("whole"), final("cut")
    assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a), "the resumed run differs from the uninterrupted one"
    print("ok  stop and resume: exit 75 on the stop request, the resumed run equals the uninterrupted one bit for bit")


def main(argv):
    known_answers()
    if "--quick" in argv:
        return 0
    tmp = tempfile.mkdtemp(prefix="toy_selftest_")
    try:
        proj = copy_of_the_problem(tmp)
        env = {**os.environ, "AR_ROOT": AR, "PYTHONPATH": AR}
        env.pop("AR_PROJECT", None)
        env.pop("TOY_DATA", None)
        chain(proj, env)
        stop_and_resume(proj, env)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("toy selftest passed")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
