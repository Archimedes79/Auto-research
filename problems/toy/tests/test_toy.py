"""The toy's own tests: known answers of data, ground truth and metric (asymmetric, so a transposed or swapped argument fails), the refusals of the
data reader, and a training stopped in the middle and resumed that equals the uninterrupted one bit for bit.

    python tests/test_toy.py [--quick]          (from the problem folder, with autoresearch/ on PYTHONPATH; --quick: no training)
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [HERE, os.path.join(os.path.dirname(os.path.dirname(HERE)), "autoresearch")]

from dataset import make_data                                       # noqa: E402
from Evaluation import FrameRef, Prediction, Sample                 # noqa: E402
from evaluation import adapter as AD                                # noqa: E402
from gt import generate as GT                                       # noqa: E402
from net.prep import shards as S                                    # noqa: E402


def known_answers(root):
    stored = {s["name"]: s["sha256"] for s in S.manifest()["shards"]}
    built = {e["name"]: e["sha256"] for e, _ in make_data.build(S.manifest()["recipe"])}
    assert built == stored, "this machine makes other data than the manifest says"
    assert make_data.make(root) == 0 and GT.generate(root)["shards"] == len(stored)
    print(GT.check(root))
    metric = AD.SegIoU()
    truth = np.zeros((4, 4), bool)
    truth[0, :3] = True                                             # 3 pixels
    pred = np.zeros((4, 4), bool)
    pred[0, :2] = True                                              # 2 of them, and one false pixel
    pred[3, 3] = True
    sample = Sample(FrameRef("s", 0), inputs=None, reference=truth)
    r = metric.reduce([metric.frame(sample, Prediction(outputs=pred))])
    assert abs(r["precision"] - 2 / 3) < 1e-12 and abs(r["recall"] - 2 / 3) < 1e-12 and abs(r["iou"] - 2 / 4) < 1e-12, r
    pred2 = np.zeros((4, 4), bool)
    pred2[0, :1] = True                                             # precision 1, recall 1/3: swapped arguments would give the reverse
    r2 = metric.reduce([metric.frame(sample, Prediction(outputs=pred2))])
    assert r2["precision"] == 1.0 and abs(r2["recall"] - 1 / 3) < 1e-12, r2
    assert metric.support(metric.frame(sample, Prediction(outputs=truth))) == {"iou": 3.0}
    print("ok  known answers: data bytes, ground truth, metric")


def refusals(root):
    name = S.split_shards("val")[0]["name"]
    raw = S.raw_path(root, name)
    keep = open(raw, "rb").read()
    try:
        with open(raw, "r+b") as f:
            f.seek(len(keep) - 4)
            f.write(b"\0\0\0\1")
        try:
            S.load_split("val", root)
        except ValueError as e:
            assert "not the shard" in str(e), e
        else:
            raise AssertionError("a changed raw shard was read")
        try:
            GT.generate(root)
        except GT.Differs:
            pass
        else:
            raise AssertionError("the ground truth was made from a changed raw shard")
    finally:
        with open(raw, "wb") as f:
            f.write(keep)
    sig = GT.signature()
    t = S.truth_path(root, sig, name)
    good = open(t, "rb").read()
    try:
        np.save(t, np.zeros((40, 32, 32), bool), allow_pickle=False)
        try:
            GT.generate(root)
        except GT.Differs:
            pass
        else:
            raise AssertionError("a damaged ground-truth shard was accepted")
        assert not GT.have(root)[0]
    finally:
        with open(t, "wb") as f:
            f.write(good)
    os.environ.pop("RESEARCH_HOLDOUT_RUN", None)
    try:
        S.holdout_guard("test")
    except PermissionError:
        pass
    else:
        raise AssertionError("the holdout split opened outside a TEST run")
    print("ok  refusals: changed raw shard, damaged ground truth, holdout split")


def stop_and_resume(root, tmp):
    from trainkit import control
    base = json.load(open(os.path.join(HERE, "training", "configs", "base.json"), encoding="utf-8"))
    base.update(steps=3000, save_every=500, eval_every=3000, log_every=1, threads=1, data_root=root, out_root=os.path.join(tmp, "runs"))
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path[:2]))

    def start(name):
        cfg = os.path.join(tmp, name + ".json")
        json.dump({**base, "name": name}, open(cfg, "w", encoding="utf-8"))
        return [sys.executable, os.path.join(HERE, "training", "train.py"), cfg]

    subprocess.run(start("whole"), cwd=HERE, env=env, check=True, capture_output=True)
    cmd = start("cut")
    proc = subprocess.Popen(cmd, cwd=HERE, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    log = os.path.join(tmp, "runs", "cut", "log.jsonl")
    while proc.poll() is None and not (os.path.exists(log) and os.path.getsize(log) > 0):
        pass
    control.request_stop(os.path.join(tmp, "runs", "cut"), "test")
    out = proc.communicate(timeout=300)[0]
    assert proc.returncode == control.EXIT_STOPPED, f"exit {proc.returncode} (a run that ended before the stop request proves nothing)\n{out[-800:]}"
    subprocess.run(cmd, cwd=HERE, env=env, check=True, capture_output=True)

    def final(name):
        ck = torch.load(os.path.join(tmp, "runs", name, "ckpt_00003000.pt"), map_location="cpu", weights_only=False)
        return {**ck["state"], **{"ema." + k: v for k, v in ck["ema"].items()}}

    a, b = final("whole"), final("cut")
    assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a), "the resumed run differs from the uninterrupted one"
    print("ok  stop and resume: exit 75 on the stop request, the resumed run equals the uninterrupted one bit for bit")


def main(argv):
    tmp = tempfile.mkdtemp(prefix="toy_test_")
    try:
        root = os.path.join(tmp, "data")
        known_answers(root)
        refusals(root)
        if "--quick" not in argv:
            stop_and_resume(root, tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("toy tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
