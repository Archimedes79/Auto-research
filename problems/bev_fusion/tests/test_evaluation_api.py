"""Gate of the class interface of the evaluation on the BEV system (`autoresearch/Evaluation/api.py`, adapter `evaluation/bev_eval.py`, command line `python -m Evaluation`).

Without option (CPU, seconds, no data): the checks of the interface on its toy project (`Evaluation/selftest_api.py`), the adapter contract (what `api.load_adapter` returns: bev_eval
defines `defaults.REQUIRED` itself, the defaults of `Evaluation/defaults.py` add name / out / network / resolve, bev_eval's own identity wins), and the command line lists its methods.
With --real [RUN] (a finished run with a checkpoint made after D76, in the site's run root `netpath.SITE/runs`, default runs/t21_ctl_s1; D81: older runs are not run any more; CPU, 4 frames,
about 5 minutes; skips with a message without a checkpoint): the proof that the interface changes no number. On a copy of the run folder (config, GT identity, code snapshot when there is one,
the checkpoint linked):
  1  `Evaluation.run(RUN, "frames:scene-0106/3:val")` = the engine's own run of the same objects (`bev_adapter.build`, `quant.run(scenes, max_frames=4, state="carry")`): every score of that
     frame equal, number for number; the scene ran whole and warm up to the frame asked for, the other three frames are not stored
  2  a second `run` takes the stored predictions (reused, same ids); the same predictions scored with the fusion metrics added give a new evaluation id and the same numbers for the
     families both have
  3  `check` (the dry run, with the id it would give) and `list` name what is stored
Without option it also tests the adapter's identity of a network (a retrained checkpoint of the same name is another identity), the run resolution, the refusal of TEST and of unknown metrics.
Removed in v2 (D154): the metric card (`bev_eval.card`, `bev_suite`, `autoresearch/tools/card.py` are gone), the problem's own Runner (`runner/runner.py`: the default
`Evaluation/netrunner.py` runs the network) and bev_eval's own `resolve` with its folding of a run id `a.b` to the folder `a_b` (the default resolve takes a path or a name in the run root).
Usage (from the problem folder): python tests/test_evaluation_api.py [--real [RUN]] [--scene scene-0106] [--index 3]
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
NEED_GB = 18
sys.path[:0] = [ROOT, os.path.join(ROOT, "..", "..", "autoresearch"), os.path.join(ROOT, "net")]


def quick():
    from Evaluation.selftest_api import run_all
    from Evaluation.api import Evaluation as _Evaluation
    r = run_all(type(_Evaluation(project=ROOT).runner))                                                    # the Evaluation together with the problem's Runner (the default one) on the toy adapter
    assert r["ok"], r
    os.chdir(ROOT)
    from evaluation import bev_eval
    from Evaluation import defaults
    from Evaluation.api import ADAPTER_FUNCTIONS, Evaluation, load_adapter
    ad = load_adapter(None, ROOT)                                                                       # the adapter evaluation.json names, with the defaults for what it does not define
    assert isinstance(ad, defaults.Adapter) and ad._mod is bev_eval, ad
    missing = [f for f in defaults.REQUIRED if not hasattr(bev_eval, f)]
    assert not missing, f"bev_eval lacks {missing}"
    missing = [f for f in ADAPTER_FUNCTIONS if not hasattr(ad, f)]
    assert not missing, f"the adapter lacks {missing}"
    assert ad.identity is bev_eval.identity and ad.sets is bev_eval.sets, "what the adapter defines itself wins over the defaults"
    assert hasattr(bev_eval, "reference") and hasattr(bev_eval, "time") and hasattr(bev_eval, "probe")
    assert hasattr(bev_eval, "free_source") and hasattr(bev_eval, "free_subject"), "the Runner's pair: both or none"
    from Evaluation.api import EvaluationError
    from evaluation import bev_adapter                                                                   # an evaluator that cannot be hashed fails closed (a renamed file of NUMBER_PATHS)
    assert "bev_eval.py" in {os.path.basename(p) for p in bev_adapter.NUMBER_PATHS}, "the adapter's own code is part of the evaluator hash (and so of the evaluation id)"
    saved_paths = bev_adapter.NUMBER_PATHS
    bev_adapter.NUMBER_PATHS = saved_paths + (os.path.join(ROOT, "evaluation", "no_such_file_renamed.py"),)
    try:
        assert bev_eval.evaluator_now() is None
    finally:
        bev_adapter.NUMBER_PATHS = saved_paths
    import netpath
    from evaluation import bev_annotation_pts                                                            # derived data is read from the site, not from the problem folder
    assert bev_annotation_pts.OUT == os.path.join(netpath.SITE, "data", "cache", "annotation_pts.npz"), bev_annotation_pts.OUT
    if os.path.exists(bev_annotation_pts.OUT):                                                           # counts read from sample_annotation.json by hand: (lidar, radar); the first token ends in "00"
        pts = bev_annotation_pts.load()
        for tok, want in (("30f65d58c9c34488991774bec0ff0500", (3, 1)), ("5bb38a1b262649debccfd4fc00c1c600", (11, 1)), ("c28b310e5a3f4f258f1ef52e0f3c3700", (1, 2))):
            assert pts(tok) == want, (tok, pts(tok), want)
    else:
        print(f"SKIP the annotation table's known counts: no {bev_annotation_pts.OUT}")
    ev_ = Evaluation(project=ROOT)                                                                     # evaluation.json names no Runner: the default one, built with (project, adapter)
    assert type(ev_.runner).__module__ == "Evaluation.netrunner", type(ev_.runner)
    assert type(Evaluation(adapter="evaluation.bev_eval").runner).__module__ == "Evaluation.netrunner", "the Evaluation always includes the project's Runner, whichever way the adapter is named"
    cfg = json.load(open(os.path.join(ROOT, "evaluation", "evaluation.json")))
    assert cfg == {"adapter": "evaluation.bev_eval"}, cfg
    s = bev_eval.sets()
    assert s["small.val"] == 10 and s["large.val"] == 82 and s["val"] is None and s["train"] is None, s
    # the identity of a network: its checkpoint (name, fingerprint), config, code and the adapter (bev_adapter.py and bev_free.py); a retrained checkpoint of the same name is another identity
    with tempfile.TemporaryDirectory() as tmp:
        run = os.path.join(tmp, "fake_run")
        os.makedirs(run)
        open(os.path.join(run, "config.json"), "w").write("{}")
        ck = os.path.join(run, "ckpt_00000010.pt")
        open(ck, "wb").write(os.urandom(50_000))
        a = bev_eval.identity(run)
        assert a["checkpoint"] == "ckpt_00000010.pt" and a["adapter"] and a == bev_eval.identity(run) == ad.identity(run), a
        file_hash = bev_eval._file_hash
        for edited in ("bev_free.py", "netrunner.py", "torch_taps.py"):                                    # an edit of the label-free path or of the frame loop / taps is another identity
            bev_eval._file_hash = lambda p, edited=edited: file_hash(p) + ("x" if p.endswith(edited) else "")
            try:
                assert bev_eval.identity(run)["adapter"] != a["adapter"], f"{edited} is part of the identity"
            finally:
                bev_eval._file_hash = file_hash
        assert a["code"].startswith("live code "), "a run without a code snapshot is identified by the live training/ and net/ code it will be imported from"
        data = bytearray(open(ck, "rb").read())
        data[25_000] ^= 1                                                                                 # a byte in the MIDDLE of the file
        open(ck, "wb").write(bytes(data))
        assert bev_eval.identity(run)["sha1"] != a["sha1"], "a retrained checkpoint of the same name and size is another identity"
        open(os.path.join(run, "config.json"), "w").write('{"seed": 1}')
        assert bev_eval.identity(run)["config"] != a["config"], "another config is another identity"
        os.makedirs(os.path.join(run, "code"))
        open(os.path.join(run, "code", "SNAPSHOT.txt"), "w").write("commit abc123 (git archive; run x)")
        assert bev_eval.identity(run)["code"] == "commit abc123 (git archive; run x)", "a snapshot names the code"
        open(os.path.join(run, "ckpt_00000020.pt"), "wb").write(bytes(data))
        assert bev_eval.identity(run)["checkpoint"] == "ckpt_00000020.pt", "the last checkpoint is the one evaluated"
        try:
            bev_eval.identity(tmp)
            raise AssertionError("identity of a folder without checkpoint")
        except FileNotFoundError as e:
            assert "no checkpoint" in str(e)
        saved_env = {k: os.environ.pop(k, None) for k in ("RESEARCH_RUNS", "RESEARCH_EVAL_OUT")}             # the defaults: a run folder by path or by name in the run root
        os.environ["RESEARCH_RUNS"] = tmp
        try:
            assert ad.resolve(run) == ad.resolve("fake_run") == os.path.abspath(run) and ad.network("fake_run") == run, (ad.resolve("fake_run"), ad.network("fake_run"))
            assert ad.name(run) == "fake_run" and ad.out(run) == os.path.join(run, "eval"), (ad.name(run), ad.out(run))
            try:
                ev_.net_identity(ev_.resolve("no_such_run_anywhere"))
                raise AssertionError("an unknown run was taken")
            except EvaluationError as e:
                assert "no checkpoint" in str(e), e
        finally:
            for k, v in saved_env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
    for bad in ("test", "small.test", "large.test"):                                                          # TEST is refused before anything is built
        try:
            bev_eval.source("runs/none", bad)
            raise AssertionError(f"{bad} was accepted")
        except ValueError as e:
            assert "TEST" in str(e), e
    from Evaluation.api import EvaluationError                                                              # one network per process: its code modules stay imported
    saved = dict(bev_eval._CODE)
    bev_eval._CODE["path"] = "/some/other/snapshot"
    try:
        bev_eval.source("runs/_no_such_run", "val")
        raise AssertionError("a second code snapshot was accepted in one process")
    except EvaluationError as e:
        assert "one network per process" in str(e), e
        bev_eval._CODE["path"] = "/some/other/snapshot"                           # every way into the network's code claims it: a task, a timing and a probe too
        for call in (lambda: bev_eval.task("runs/_no_such_run", ("maps", "boxes")), lambda: bev_eval.time("runs/_no_such_run"), lambda: bev_eval.probe("runs/_no_such_run")):
            try:
                call()
                raise AssertionError("a second code snapshot was accepted by a task / timing / probe")
            except EvaluationError as e:
                assert "one network per process" in str(e), e
    finally:
        bev_eval._CODE.update(saved)
    import torch                                                                                         # cuda asked for: moved when there, an error (never a quiet CPU run) when not
    from types import SimpleNamespace
    moved = []
    box = SimpleNamespace(to=lambda d: moved.append(d))
    fake = SimpleNamespace(_bev_subject=SimpleNamespace(device="cpu", model=box, loss=box))
    real_avail = torch.cuda.is_available
    try:
        torch.cuda.is_available = lambda: False
        try:
            bev_eval.subject("runs/x", fake, "cuda")
            raise AssertionError("cuda was not there and no error")
        except EvaluationError as e:
            assert "CUDA is not available" in str(e) and moved == [], e
        torch.cuda.is_available = lambda: True
        got = bev_eval.subject("runs/x", fake, "cuda")
        assert moved == ["cuda", "cuda"] and got.device == "cuda" and fake._bev_subject is None, moved
        try:
            bev_eval.subject("runs/x", fake, "cuda")
            raise AssertionError("a Subject was handed out twice")
        except RuntimeError as e:
            assert "handed out" in str(e)
    finally:
        torch.cuda.is_available = real_avail
    try:
        bev_eval.task("runs/none", ("maps", "mapz"))
        raise AssertionError("an unknown metric was accepted")
    except ValueError as e:
        assert "unknown metrics" in str(e)
    out = subprocess.run([sys.executable, "-m", "Evaluation", "methods"], cwd=os.path.join(ROOT, "..", "..", "autoresearch"), capture_output=True, text=True, timeout=120)
    assert out.returncode == 0 and "Evaluation.run(" in out.stdout and "Evaluation.report(" in out.stdout, out.stderr[-300:]
    print(f"quick: interface checks {r['checks']} on the toy project, bev_eval has the adapter contract, sets {len(s)}, the command line lists its methods")


def available_gb():
    try:
        with open("/proc/meminfo") as f:
            return int(next(ln for ln in f if ln.startswith("MemAvailable")).split()[1]) / 1e6
    except (OSError, StopIteration, ValueError):
        return None


def real(run, scene, index):
    import netpath
    src_run = os.path.join(netpath.SITE, run)                                           # the site holds the link runs -> the problem's run root
    if not glob.glob(os.path.join(src_run, "ckpt_*.pt")):
        print(f"SKIP real check: {src_run} has no checkpoint here")
        return False
    free = available_gb()
    if free is not None and free < NEED_GB:                                          # the box is shared: the real check holds two built networks and their tables (about 15 GB at its peak)
        print(f"SKIP real check: {free:.1f} GB available, it needs {NEED_GB} GB (wait until the other jobs are done)")
        return False
    os.chdir(ROOT)
    from evaluation import bev_adapter as A
    from Evaluation import EvalSuite
    from Evaluation.api import Evaluation
    with tempfile.TemporaryDirectory() as tmp:
        net = os.path.join(tmp, os.path.basename(run))                                  # absolute: the adapter works in the site folder
        os.makedirs(net)
        for f in ("config.json", "gt.json"):
            if os.path.isfile(os.path.join(src_run, f)):
                shutil.copy(os.path.join(src_run, f), net)
        if os.path.isdir(os.path.join(src_run, "code")):
            shutil.copytree(os.path.join(src_run, "code"), os.path.join(net, "code"))
        ck = sorted(glob.glob(os.path.join(src_run, "ckpt_*.pt")))[-1]
        os.symlink(ck, os.path.join(net, os.path.basename(ck)))
        frames = f"frames:{scene}/{index}:val"
        ev = Evaluation(project=ROOT)                                                                     # the default Runner (Evaluation/netrunner.py) predicts, the Evaluation scores
        assert type(ev.runner).__module__ == "Evaluation.netrunner", type(ev.runner)
        # 1 the interface against the engine's own run of the same objects
        r = ev.run(net, frames, gt_key=None, device="cpu", workers=0, progress=0)
        assert r["n_frames"] == 1 and r["frames_predicted"] == 1 and r["predictions_reused"] is False and r["warnings"] == [], r
        rs_dir = os.path.join(ev.ad.out(net), "results", r["results"].split("@")[1])
        assert len(os.listdir(os.path.join(rs_dir, "frames"))) == 1, "only the frame asked for is stored (the scene ran whole up to it)"
        from Evaluation.results import ResultSet
        rs0 = ResultSet.open(rs_dir)
        stored0 = rs0.load(rs0.refs()[0])
        assert "loss" not in stored0.info and not any(k.startswith("group.") for k in stored0.info) and stored0.info.get("device"), "the Runner read no ground truth: no validation loss in what it stored"
        assert rs0.manifest["identity"]["inputs"].get("dataset") and "gt_signature" not in rs0.manifest["identity"]["inputs"], rs0.manifest["identity"]["inputs"]
        src, subj, task = A.build(net, device="cpu", split="val", workers=0)
        task.fusion = True                                                                                 # D31: the Evaluation scores the fusion area by default, so does the engine run it is compared with
        suite = EvalSuite(src, subj, task, out=os.path.join(tmp, "engine"), state="carry")
        store = suite.quant.run(scenes=[scene], max_frames=index + 1, name="engine", progress=0)
        mine = ev.split_numbers(r["id"], "frame")
        want = {k: v for k, v in next(fr["scores"] for fr in store.frames if fr["label"] == f"{scene}/{index:03d}").items() if not k.startswith("loss.")}    # the Runner reads no ground truth: no val loss
        assert set(mine) == {f"{scene}/{index:03d}"}, set(mine)
        got = mine[f"{scene}/{index:03d}"]
        bad = [k for k in want if k in got and abs(got[k] - want[k]) > 1e-9 * max(1.0, abs(want[k]))]
        assert set(want) == set(got) and not bad, f"scores differ: {bad[:5]}"
        suite.close()
        print(f"  1 ok: {len(got)} scores of {scene}/{index} equal the engine's run (scene run whole and warm to the frame, one frame stored)")
        # 2 reuse and rescoring
        r2 = ev.run(net, frames, device="cpu", workers=0, progress=0)
        assert r2["predictions_reused"] and r2["id"] == r["id"], r2
        r3 = ev.run(net, frames, metrics=("maps", "boxes", "curves"), device="cpu", workers=0, progress=0)              # D31: the fusion-area scoring is always on, so the family added here is the curves
        assert r3["predictions_reused"] and r3["results"] == r["results"] and r3["id"] != r["id"], r3
        q2, q3 = ev.show_numbers(r["id"])["numbers"], ev.show_numbers(r3["id"])["numbers"]
        shared = [k for k in q2 if k in q3]
        assert any("detf" in k for k in q2), "the fusion-area numbers are in the default scoring (D31)"
        assert shared and all(abs(q2[k] - q3[k]) <= 1e-12 * max(1.0, abs(q2[k])) or (q2[k] != q2[k] and q3[k] != q3[k]) for k in shared) and len(q3) > len(q2), "curves added, the rest unchanged"
        print(f"  2 ok: the stored predictions were reused; the curve numbers added: {len(q3) - len(q2)} more, {len(shared)} shared numbers equal (the fusion-area numbers are in both)")
        # 3 check, list
        chk = ev.check(net, frames, workers=0)
        assert chk["ok"] and chk["frames_stored"] == 1 and chk["frames_to_run"] == 0 and chk["network_runs"] is False and chk["id"] == r["id"], chk
        assert {x["kind"] for x in ev.list(net)} == {"predictions", "evaluation"} and len(ev.list(net)) == 3
        print("  3 ok: check, list")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", nargs="?", const="runs/t21_ctl_s1", default=None, metavar="RUN")
    ap.add_argument("--scene", default="scene-0106")
    ap.add_argument("--index", type=int, default=3)
    a = ap.parse_args()
    quick()
    done = "interface, adapter contract, command line ok"
    if a.real:
        done += "; " + ("real check ok" if real(a.real, a.scene, a.index) else "real check skipped")
    print(f"evaluation api: {done}")


if __name__ == "__main__":
    main()
