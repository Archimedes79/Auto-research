"""Gate of the evaluation's front door `evaluation/evaluate.py` (plan 2026-11-03, section 1: one command for the evaluation).

Without option (CPU, seconds, no data): the door only translates options into `bev_suite.py` commands and runs them, so
1  the plan of `--all` is the evaluation chain of the t16 bring-up (command by command, in its order, with its constants) and then the report page, and every single
   option gives the command its table says (eval --n N, qual random / frame, qual videos N / --scenes, timing + time); the TEST split is not offered; a bad value is refused
2  the executor: a failed fatal step ends the call (exit 1), a failed soft step of `--all` is listed and the chain goes on (exit 0), generalisation gets a second try
   and its half-made store is moved out of quant/ (never deleted), `RUN/eval/door.json` holds every call with the commands and their exit codes
With --real (a finished run, default runs/t16_full_s0, CPU, 4 frames, about 2 minutes: two bev_suite processes load nuScenes and the model): the golden check, the
proof that the door changes nothing: `evaluate.py RUN --quant --n N` and `bev_suite.py RUN eval --n N` on two copies of the run (config, gt.json, code snapshot; the
checkpoint through --ckpt-dir, the stored eval/ of the run is never written) give the same metrics.json, card.json (metrics, directions, protocol, frames) and table.txt,
number for number. CPU, because a GPU is not bit-reproducible: two runs of the same bev_suite command differ in about 800 of 5200 numbers (median 1e-5 relative, a
detection at the score threshold in or out: n_det +-1), so on the GPU the door is only inside that noise. It skips with a message when the run has no checkpoint here.

Usage (from the bev_fusion folder): python tests/test_evaluate_door.py [--real [RUN] [--n 4] [--device cpu|cuda]]
"""
import argparse
import contextlib
import glob
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import evaluate as E  # noqa: E402

G6 = ["--device", "cuda", "--workers", "2", "--split", "val"]                    # the global options the door writes out (the defaults of bev_suite)


def steps_of(*argv):
    return E.plan(E.build_parser().parse_args(["runs/x", *argv]))


def cmds(steps):
    """name -> the arguments after the global options (the door writes G6 first; a step with its own --out has it right behind)"""
    out = {}
    for s in steps:
        argv = s["argv"]
        assert argv is None or argv[:6] == G6, argv
        out[s["name"]] = None if argv is None else argv[6:]
    return out


def refused(*argv):
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            steps_of(*argv)
    except SystemExit:
        return True
    return False


def check_plan():
    # 1  --all = temp/t16/eval_chain.sh, command by command, in its order, plus the report page as the last step (the chain made it inside generalisation, before the videos)
    allp = steps_of("--all")
    assert [s["name"] for s in allp] == ["timing", "numbers", "generalisation", "modules", "videos", "showcase", "report"], [s["name"] for s in allp]
    c = cmds(allp)
    assert c["timing"] == ["timing", "--n", "30"]
    assert c["numbers"] == ["--curves", "--fusion", "--velocity", "eval", "--n", "1000", "--prior", "500", "--images", "8"]
    assert c["generalisation"] == ["generalisation", "--n", "1000"]
    assert c["modules"] == ["modules", "--frames", "4"]                              # the chain's `--device cuda` is the global option
    assert c["videos"] == ["qual", "videos", "3", "--seed", "1"]
    assert c["showcase"] == ["--out", "runs/x/eval/showcase", "qual", "videos", "6", "--view", "overview", "--seed", "21"]
    assert c["report"] == ["report", "--no-pictures", "--val-table", "runs/x/eval/by_n/table_n1000.txt", "--train-table", "runs/x/eval/by_n/table_train_n1000.txt"], \
        "the tables generalisation passes to its page"
    assert [s["fatal"] for s in allp] == [False, True, False, False, False, False, False], "in the chain only the numbers are fatal"
    assert allp[2]["tries"] == 2 and allp[2]["store"] == "runs/x/eval/suite/quant/eval_train_n1000"
    plus = steps_of("--all", "--qual", "scene-0003/1", "--videos", "scene-0003")           # more asked for on top of the chain: soft too, and where they belong
    assert [(s["name"], s["fatal"]) for s in plus] == [("timing", False), ("numbers", True), ("generalisation", False), ("modules", False), ("frames", False),
                                                       ("videos", False), ("showcase", False), ("report", False)]
    assert [(s["name"], s["fatal"]) for s in steps_of("--all", "--qual", "2")][4] == ("qual", False)
    assert cmds(steps_of("--all", "--n", "200", "--out", "o"))["showcase"][:2] == ["--out", "o/showcase"]
    # every single option
    assert cmds(steps_of("--quant", "--n", "50")) == {"numbers": ["eval", "--n", "50"]}, "the golden relation: --quant is a plain eval"
    assert cmds(steps_of("--quant", "--curves", "--velocity"))["numbers"] == ["--curves", "--velocity", "eval", "--n", "1000"]
    assert cmds(steps_of("--time")) == {"time": ["time"], "timing": ["timing", "--n", "30"]}
    # --out: bev_timing's is a FILE (bev_suite hands its global --out on to it), so timing gets a file inside the folder; the rest the folder, the showcase a subfolder
    assert cmds(steps_of("--time", "--out", "o")) == {"timing": ["--out", "o/timing_modules.json", "timing", "--n", "30"], "time": ["--out", "o", "time"]}
    c = cmds(steps_of("--all", "--out", "temp/o"))
    assert c["timing"][:2] == ["--out", "temp/o/timing_modules.json"] and c["numbers"][:2] == ["--out", "temp/o"] and c["showcase"][:2] == ["--out", "temp/o/showcase"]
    assert c["generalisation"][:2] == ["--out", "temp/o"] and c["modules"][:2] == ["--out", "temp/o"] and c["report"][:2] == ["--out", "temp/o"]
    assert all(s["fatal"] for s in steps_of("--time", "--videos", "2")), "asked for by name: fatal"
    assert cmds(steps_of("--qual", "4", "--seed", "7")) == {"qual": ["qual", "random", "4", "--seed", "7"]}
    fr = steps_of("--qual", "scene-0061/12,scene-0003/5")
    assert [(s["name"], s["argv"], s["frames"]) for s in fr] == [("frames", None, ["scene-0061/12", "scene-0003/5"])]
    assert cmds(steps_of("--videos", "scene-0003,scene-0012")) == {"videos": ["qual", "videos", "--scenes", "scene-0003", "scene-0012", "--seed", "1"]}
    assert cmds(steps_of("--videos", "2")) == {"videos": ["qual", "videos", "2", "--seed", "1"]}
    assert [s["name"] for s in steps_of("--videos", "1", "--quant", "--time")] == ["timing", "time", "numbers", "videos"], "timing first (idle GPU)"
    g = steps_of("--quant", "--device", "cpu", "--workers", "0", "--split", "train", "--ckpt-dir", "d", "--out", "o")[0]["argv"]
    assert g[:-3] == ["--device", "cpu", "--workers", "0", "--split", "train", "--ckpt-dir", "d", "--out", "o"], g
    # refusals: no TEST split, no --all on train, malformed names
    assert refused("--quant", "--split", "test") and refused("--all", "--split", "train")
    assert refused("--qual", "scene-0061") and refused("--videos", "scene-0061/12") and refused("--qual", "0") and refused("--videos", ",")
    assert refused("--quant", "--n", "0") and refused("--all", "--n", "0") and refused("--quant", "--n", "-5") and refused("--quant", "--n", "x")
    with contextlib.redirect_stdout(io.StringIO()) as out:                           # no option: the help, exit 2
        assert E.main(["runs/x"]) == 2 and E.main([]) == 2
    assert "--quant" in out.getvalue() and "--all" in out.getvalue()


def fake_runner(codes):
    """runner(a, step) -> the next exit code listed for the step (default 0); records the names called"""
    seen = []

    def runner(a, st):
        seen.append(st["name"])
        return (codes.get(st["name"]) or [0]).pop(0)
    return runner, seen


def check_execute():
    with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()) as out:
        run = os.path.join(tmp, "runs", "x")
        a = E.build_parser().parse_args([run, "--all"])
        a.run = run
        # all pass
        runner, seen = fake_runner({})
        assert E.execute(a, E.plan(a), runner) == 0 and seen == ["timing", "numbers", "generalisation", "modules", "videos", "showcase", "report"]
        assert "all steps passed" in out.getvalue()
        # the numbers fail: fatal, nothing after them runs, the call ends with the code of the step
        runner, seen = fake_runner({"numbers": [3]})
        assert E.execute(a, E.plan(a), runner) == 3 and seen == ["timing", "numbers"]
        # timing, generalisation (twice) and modules fail: listed, the chain goes on, and the call still ends non-zero (the first failure's code);
        # the half-made store is set aside, not deleted
        store = os.path.join(run, "eval", "suite", "quant", "eval_train_n1000")
        os.makedirs(store)
        with open(os.path.join(store, "frames.jsonl"), "w") as f:
            f.write("half\n")
        runner, seen = fake_runner({"timing": [2], "generalisation": [1, 1], "modules": [1]})
        assert E.execute(a, E.plan(a), runner) == 2
        assert seen == ["timing", "numbers", "generalisation", "generalisation", "modules", "videos", "showcase", "report"], seen
        assert not os.path.exists(store)
        kept = glob.glob(os.path.join(run, "eval", "suite", "failed_stores", "eval_train_n1000.*", "frames.jsonl"))
        assert len(kept) == 1 and open(kept[0]).read() == "half\n"
        # generalisation passes at the second try: nothing failed
        runner, seen = fake_runner({"generalisation": [1, 0]})
        assert E.execute(a, E.plan(a), runner) == 0 and seen.count("generalisation") == 2
        # asked for by name (not --all): a failed timing is fatal and `time` does not run after it
        b = E.build_parser().parse_args([run, "--time"])
        runner, seen = fake_runner({"timing": [1]})
        assert E.execute(b, E.plan(b), runner) == 1 and seen == ["timing"]
        # a soft step that gives no usable code (killed: -9, or 256) still fails the call, with 1
        for bad in (-9, 256):
            runner, seen = fake_runner({"modules": [bad]})
            assert E.execute(a, E.plan(a), runner) == 1 and seen[-1] == "report", bad
        # door.json: every call, with the commands and exit codes
        calls = json.load(open(os.path.join(run, "eval", "door.json")))["calls"]
        assert len(calls) == 7, len(calls)
        assert [c["exit"] for c in calls] == [0, 3, 2, 0, 1, 1, 1] and calls[1]["failed"] == ["numbers"] and calls[2]["failed"] == ["timing", "generalisation", "modules"]
        last = calls[4]
        assert last["options"]["time"] is True and [s["name"] for s in last["steps"]] == ["timing"]
        assert last["steps"][0]["command"] == ["python", "evaluation/bev_suite.py", run] + G6 + ["timing", "--n", "30"] and last["steps"][0]["exit"] == 1
        assert calls[2]["steps"][2]["tries"] == 2 and calls[2]["steps"][2]["exit"] == 1
        text = out.getvalue()                                                       # one summary line says what failed and what happened to the chain
        assert "FAILED numbers (exit 3); the chain stopped there; exit code 3" in text
        assert "FAILED timing (exit 2), generalisation (exit 1), modules (exit 1); the chain went on, these steps are not fatal; exit code 2" in text


def check_refusals():
    """the call is refused before any step: no checkpoint, outputs that would be overwritten (a stand-in run folder, never a real run), names that are not in the val list"""
    def ns(*argv):
        a = E.build_parser().parse_args([run, *argv])
        a.run = run
        return a

    def lister(a):
        calls.append(a.device)
        return {"scene-0003": 38, "scene-0012": 38}
    with tempfile.TemporaryDirectory() as tmp:
        run, calls = os.path.join(tmp, "runs", "x"), []
        os.makedirs(os.path.join(run, "eval"))
        assert "not a run folder" in E.refusal(ns("--quant"), lister)
        open(os.path.join(run, "config.json"), "w").write("{}")
        assert "no checkpoint" in E.refusal(ns("--quant"), lister) and "--ckpt-dir" in E.refusal(ns("--quant"), lister)
        moved = os.path.join(tmp, "moved")
        assert "is not a folder" in E.refusal(ns("--quant", "--ckpt-dir", moved), lister)
        os.makedirs(moved)
        assert "no checkpoint" in E.refusal(ns("--quant", "--ckpt-dir", moved), lister)
        open(os.path.join(moved, "ckpt_00000001.pt"), "w").write("x")
        assert E.refusal(ns("--quant", "--ckpt-dir", moved), lister) is None, "a moved checkpoint is found"
        open(os.path.join(run, "ckpt_00000001.pt"), "w").write("x")
        for opts in (["--quant"], ["--time"], ["--all"], ["--videos", "2"], ["--qual", "3"], ["--quant", "--split", "train"]):
            assert E.refusal(ns(*opts), lister) is None, opts                       # a run without outputs: go on
        ev = os.path.join(run, "eval")
        # outputs that exist are not rewritten without --replace; the message names the file; what the option does not write does not block it
        for exists, opts, blocked in (("card.json", ["--quant"], True), ("card.json", ["--all"], True), ("card.json", ["--time"], False), ("card.json", ["--quant", "--split", "train"], False),
                                      ("card_train.json", ["--quant", "--split", "train"], True), ("card.json", ["--videos", "2"], False),
                                      ("timing_modules.json", ["--time"], True), ("timing_modules.json", ["--all"], True), ("timing_modules.json", ["--quant"], False),
                                      ("latency.json", ["--time"], True), ("latency.json", ["--all"], False), ("latency.json", ["--quant"], False)):
            path = os.path.join(ev, exists)
            open(path, "w").write("{}")
            why = E.refusal(ns(*opts), lister)
            assert (why is not None) == blocked and (not blocked or (exists in why and "--replace" in why)), (exists, opts, why)
            assert E.refusal(ns(*opts, "--replace"), lister) is None, (exists, opts, "--replace goes on")
            os.remove(path)
        out_dir = os.path.join(tmp, "o")                                            # with --out the timing file lives in the folder
        os.makedirs(out_dir)
        open(os.path.join(ev, "timing_modules.json"), "w").write("{}")
        assert E.refusal(ns("--time", "--out", out_dir), lister) is None, "the run's own file is not what --out writes"
        open(os.path.join(out_dir, "timing_modules.json"), "w").write("{}")
        assert "timing_modules.json" in E.refusal(ns("--time", "--out", out_dir), lister)
        os.remove(os.path.join(ev, "timing_modules.json"))
        os.remove(os.path.join(out_dir, "timing_modules.json"))
        # names: random picks need no list; named scenes / frames must be in it (a frame index below the number of frames)
        assert E.refusal(ns("--qual", "3", "--videos", "2"), lister) is None and calls == []
        for opts, ok in ((["--videos", "scene-0003,scene-0012"], True), (["--videos", "scene-0003,scene-9999"], False), (["--qual", "scene-0003/37,scene-0012/0"], True),
                         (["--qual", "scene-0003/38"], False), (["--qual", "scene-0003/x"], False), (["--qual", "scene-9999/1"], False)):
            why = E.refusal(ns(*opts), lister)
            assert (why is None) == ok and (ok or opts[1].split(",")[-1].split("/")[0] in why or "frames" in why), (opts, why)
        asked, real_run = [], E.subprocess.run                                      # the real lister: `bev_suite.py RUN list` on the CPU (it takes no GPU from a job), "scene frames" lines
        E.subprocess.run = lambda cmd, **kw: asked.append(cmd) or subprocess.CompletedProcess(cmd, 0, stdout="scene-0003 38\nsome warning\nscene-0012 40\n", stderr="")
        try:
            assert E.list_scenes(ns("--videos", "scene-0003", "--device", "cuda", "--workers", "2")) == {"scene-0003": 38, "scene-0012": 40}
        finally:
            E.subprocess.run = real_run
        assert asked[0][1:3] == [E.SUITE, run] and asked[0][3:9] == ["--device", "cpu", "--workers", "0", "--split", "val"] and asked[0][-1] == "list", asked
        assert "no scenes" in E.refusal(ns("--videos", "scene-0003"), lambda a: {})
        # through main: refused = exit 2 with the reason on stderr, nothing run; --replace goes on to execute
        ran, real_execute = [], E.execute
        E.execute = lambda a, steps: ran.append([s["name"] for s in steps]) or 0
        try:
            open(os.path.join(ev, "card.json"), "w").write("{}")
            with contextlib.redirect_stderr(io.StringIO()) as err:
                assert E.main([run, "--quant"]) == 2 and ran == []
            assert "refused" in err.getvalue() and "card.json" in err.getvalue() and "--replace" in err.getvalue(), err.getvalue()
            assert E.main([run, "--quant", "--replace"]) == 0 and ran == [["numbers"]]
            with contextlib.redirect_stderr(io.StringIO()):
                assert E.main([run, "--qual", "scene-0061"]) == 2 and ran == [["numbers"]], "a malformed name is refused too"
        finally:
            E.execute = real_execute


def same(a, b, what):
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True), f"{what} differs"


def check_golden(run, n, device):
    """evaluate.py --quant against bev_suite.py eval on two copies of the run; every number must be the same"""
    if not glob.glob(os.path.join(run, "ckpt_*.pt")) or not os.path.isfile(os.path.join(run, "config.json")):
        print(f"SKIP golden check: {run} has no checkpoint here")
        return False
    py = sys.executable
    with tempfile.TemporaryDirectory(dir=os.path.join(ROOT, "temp")) as tmp:
        copy = {}
        for who in ("door", "suite"):                                                # the same folder name: the card carries the run's name
            copy[who] = os.path.relpath(os.path.join(tmp, who, os.path.basename(run)), ROOT)
            os.makedirs(copy[who])
            for f in ("config.json", "gt.json"):
                if os.path.isfile(os.path.join(run, f)):
                    shutil.copy(os.path.join(run, f), copy[who])
            shutil.copytree(os.path.join(run, "code"), os.path.join(copy[who], "code"))
        common, env = ["--n", str(n), "--device", device, "--workers", "0", "--ckpt-dir", run], dict(os.environ, OMP_NUM_THREADS="4")
        logs = {who: open(os.path.join(tmp, f"{who}.log"), "w") for who in copy}
        procs = [subprocess.Popen([py, "evaluation/evaluate.py", copy["door"], "--quant"] + common + ["--out", os.path.join(copy["door"], "suite")],
                                  cwd=ROOT, env=env, stdout=logs["door"], stderr=subprocess.STDOUT),
                 subprocess.Popen([py, "evaluation/bev_suite.py", copy["suite"], "eval"] + common + ["--out", os.path.join(copy["suite"], "suite")],
                                  cwd=ROOT, env=env, stdout=logs["suite"], stderr=subprocess.STDOUT)]
        codes = [p.wait() for p in procs]
        for f in logs.values():
            f.close()
        if codes != [0, 0]:
            for who in copy:
                print(f"--- {who}.log (tail)\n" + "".join(open(os.path.join(tmp, f"{who}.log")).readlines()[-15:]))
        assert codes == [0, 0], f"exit codes door / bev_suite: {codes}"
        ev = {who: os.path.join(ROOT, copy[who], "eval") for who in copy}
        m = {who: json.load(open(os.path.join(ev[who], "metrics.json"))) for who in ev}
        c = {who: json.load(open(os.path.join(ev[who], "card.json"))) for who in ev}
        assert len(m["door"]["metrics"]) > 1000, len(m["door"]["metrics"])
        for who in ev:
            m[who].pop("run"), m[who].pop("code")                                    # the paths of the copy
        same(m["door"], m["suite"], "metrics.json")
        for who in ev:
            for k in ("date", "provenance"):                                         # a minute stamp; paths, times and environment of the process
                c[who].pop(k)
        same(c["door"], c["suite"], "card.json")
        assert open(os.path.join(ev["door"], "table.txt")).read() == open(os.path.join(ev["suite"], "table.txt")).read(), "table.txt differs"
        door = json.load(open(os.path.join(ev["door"], "door.json")))["calls"]
        assert len(door) == 1 and door[0]["exit"] == 0 and door[0]["steps"][0]["command"][1] == "evaluation/bev_suite.py" and door[0]["options"]["quant"] is True
        print(f"golden: {len(m['door']['metrics'])} numbers, card and table equal ({n} {m['door']['split']} frames, {device}, step {m['door']['metrics'].get('info/step')})")
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--real", nargs="?", const="runs/t16_full_s0", default=None, metavar="RUN", help="also the golden check on this run")
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--device", default="cpu")
    a = p.parse_args()
    os.chdir(ROOT)
    check_plan()
    check_execute()
    check_refusals()
    done = "plan, executor and refusals ok"
    if a.real:
        done += "; " + ("golden check ok" if check_golden(a.real, a.n, a.device) else "golden check skipped")
    print(f"evaluate door: {done}")


if __name__ == "__main__":
    main()
