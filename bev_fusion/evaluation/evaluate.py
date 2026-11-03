"""The one front door of the BEV evaluation: `python evaluation/evaluate.py RUN [options]` (from the bev_fusion folder, RUN = runs/<name>).

It computes nothing itself. Every option is one or more commands of `bev_suite.py`, each run as its own process, so every output is where bev_suite
puts it. What the door adds is `RUN/eval/door.json`: the options and the commands (exit code, seconds) of each call.

    --quant              eval --n N                       the numbers: table, metrics.json, card.json (+ --curves --fusion --velocity: PR / ROC, fusion area, velocity field)
    --qual N | F1,F2,..  qual random N --seed S           N random frames as pictures, or the named frames (scene-0061/12,...; one process for all of them)
    --videos N | V1,V2   qual videos N --seed S           N random scenes as videos, or the named scenes (scene-0061,...)
    --time               timing --n 30 + time             ms per module and ms per frame (idle GPU only: timing refuses a busy one)
    --all                what the evaluation chain of the t16 bring-up did: timing, the numbers with --curves --fusion --velocity --prior 500 --images 8,
                         generalisation (train against val), the module probe, 3 videos, 6 showcase videos, and the report page last (with the train-against-val
                         table and the videos; the chain made it inside generalisation, before the videos). Only the numbers are fatal there, a failed step is
                         listed and the chain goes on; generalisation gets a second try
Global options as in bev_suite: --device cuda|cpu  --workers N  --split val|train  --ckpt-dir DIR  --out DIR. The TEST split is not offered.
Nothing asked for (no option): this help, exit 2.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
SUITE = os.path.join(ROOT, "evaluation", "bev_suite.py")
TIMING_N, ALL_PRIOR, ALL_IMAGES, ALL_VIDEOS, SHOWCASE_VIDEOS, SHOWCASE_SEED = 30, 500, 8, 3, 6, 21      # the constants of the t16 chain


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, allow_abbrev=False)
    p.add_argument("run", nargs="?", help="runs/<name>")
    p.add_argument("--quant", action="store_true", help="the numbers (bev_suite eval)")
    p.add_argument("--qual", metavar="N|F1,F2", help="N random frames, or the named frames")
    p.add_argument("--videos", metavar="N|V1,V2", help="N random scenes, or the named scenes, as videos")
    p.add_argument("--time", action="store_true", help="ms per frame and per module (idle GPU only)")
    p.add_argument("--all", action="store_true", help="the whole chain of a finished run (see above)")
    p.add_argument("--n", type=int, default=1000, help="val frames of --quant / --all (default 1000)")
    p.add_argument("--seed", type=int, default=1, help="seed of the random frames and scenes (default 1)")
    p.add_argument("--curves", action="store_true", help="with --quant: also PR / ROC / calibration (bev_suite --curves)")
    p.add_argument("--fusion", action="store_true", help="with --quant: also the fusion-area scoring (bev_suite --fusion)")
    p.add_argument("--velocity", action="store_true", help="with --quant: also the velocity field (bev_suite --velocity)")
    p.add_argument("--device", default="cuda", help="cuda | cpu (default cuda)")
    p.add_argument("--workers", type=int, default=2, help="read-ahead processes (default 2)")
    p.add_argument("--split", default="val", choices=["val", "train"])
    p.add_argument("--ckpt-dir", default=None, help="the checkpoints of a run folder that has none")
    p.add_argument("--out", default=None, help="the suite folder (default RUN/eval/suite)")
    return p


def count_or_list(text: str, what: str):
    """'3' -> 3 (random picks), 'a,b' -> ['a', 'b'] (named ones); a frame id has a '/', a scene name has none"""
    if text.isdigit():
        if int(text) < 1:
            raise SystemExit(f"evaluate: --{what} {text}: at least 1")
        return int(text)
    names = [x.strip() for x in text.split(",") if x.strip()]
    if not names or any(("/" in x) != (what == "qual") for x in names):
        raise SystemExit(f"evaluate: --{what} {text!r}: a number or a comma list of " + ("frames like scene-0061/12" if what == "qual" else "scenes like scene-0061"))
    return names


def step(name, argv=None, fatal=True, tries=1, store=None, frames=None) -> dict:
    """one command: the arguments after RUN of a bev_suite call, or (frames) the named frames drawn in this process"""
    return {"name": name, "argv": argv, "fatal": fatal, "tries": tries, "store": store, "frames": frames}


def plan(a) -> list:
    """the options -> the ordered steps (timing first: it needs an idle GPU). Pure: nothing is run here"""
    if a.all and a.split == "train":
        raise SystemExit("evaluate: --all with --split train: generalisation already evaluates train; use --quant --split train")

    def global_options(out=a.out):                                                  # the global options of bev_suite, always written out
        return (["--device", a.device, "--workers", str(a.workers), "--split", a.split] + (["--ckpt-dir", a.ckpt_dir] if a.ckpt_dir else [])
                + (["--out", out] if out else []))
    G, soft = global_options(), a.all                                               # soft: in the chain only the numbers are fatal
    steps = []
    if a.time or a.all:                                                             # first, and before `time`: it refuses a GPU that is not idle, `time` only records the load
        steps.append(step("timing", G + ["timing", "--n", str(TIMING_N)], fatal=not soft))
    if a.time:
        steps.append(step("time", G + ["time"]))
    if a.quant or a.all:
        more = [f"--{f}" for f in ("curves", "fusion", "velocity") if a.all or getattr(a, f)]
        steps.append(step("numbers", G + more + ["eval", "--n", str(a.n)] + (["--prior", str(ALL_PRIOR), "--images", str(ALL_IMAGES)] if a.all else [])))
    if a.all:
        suite_out = a.out or os.path.join(a.run, "eval", "suite")
        steps.append(step("generalisation", G + ["generalisation", "--n", str(a.n)], fatal=False, tries=2, store=os.path.join(suite_out, "quant", f"eval_train_n{a.n}")))
        steps.append(step("modules", G + ["modules", "--frames", "4"], fatal=False))
    if a.qual:
        q = count_or_list(a.qual, "qual")
        steps.append(step("qual", G + ["qual", "random", str(q), "--seed", str(a.seed)], fatal=not soft) if isinstance(q, int) else step("frames", frames=q, fatal=not soft))
    if a.videos or a.all:
        v = count_or_list(a.videos, "videos") if a.videos else ALL_VIDEOS
        steps.append(step("videos", G + ["qual", "videos"] + (["--scenes"] + v if isinstance(v, list) else [str(v)]) + ["--seed", str(a.seed)], fatal=not soft))
    if a.all:
        show = os.path.join(a.out, "showcase") if a.out else os.path.join(a.run, "eval", "showcase")
        steps.append(step("showcase", global_options(show) + ["qual", "videos", str(SHOWCASE_VIDEOS), "--view", "overview", "--seed", str(SHOWCASE_SEED)], fatal=False))
        by_n = os.path.join(a.run, "eval", "by_n")                                  # the page last, so that it shows the videos; generalisation made it once without them
        steps.append(step("report", G + ["report", "--no-pictures", "--val-table", os.path.join(by_n, f"table_n{a.n}.txt"),
                                         "--train-table", os.path.join(by_n, f"table_train_n{a.n}.txt")], fatal=False))
    return steps


def suite_command(a, st) -> list:
    return [sys.executable, SUITE, a.run] + st["argv"]


def run_frames(a, specs) -> int:
    """the named frames as pictures, the model built once (a bev_suite call per frame would load it for every frame)"""
    sys.path.insert(0, ROOT)
    try:
        from evaluation import bev_suite
        from evalsuite.cli import _qual_print
        suite = bev_suite.make_suite(argparse.Namespace(run=a.run, device=a.device, workers=a.workers, split=a.split, ckpt_dir=a.ckpt_dir, control="", out=a.out, state="carry"))
        try:
            for spec in specs:
                _qual_print(suite.qual.frame(spec, ("overview",)))
        finally:
            suite.close()
        return 0
    except (Exception, SystemExit) as ex:
        print(f"frames failed: {type(ex).__name__}: {ex}", flush=True)
        return 1


def run_step(a, st) -> int:
    return run_frames(a, st["frames"]) if st["frames"] else subprocess.call(suite_command(a, st))


def set_aside(store: str) -> None:
    """a half-made store of a failed try is moved out of quant/ (the report page reads quant/*), never deleted"""
    if os.path.isdir(store):
        dest = os.path.join(os.path.dirname(os.path.dirname(store)), "failed_stores", f"{os.path.basename(store)}.{time.strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        os.replace(store, dest)
        print(f"moved the half-made store to {dest}", flush=True)


def execute(a, steps, runner=run_step) -> int:
    """run the steps in order; a failed fatal step ends the call (exit 1), a failed soft one is listed and the call goes on. RUN/eval/door.json
    = {"calls": [...]}: the earlier calls and this one, rewritten after each step so that a crash leaves its trail"""
    path = os.path.join(a.run, "eval", "door.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        with open(path) as f:
            earlier = json.load(f)["calls"]
    except (OSError, ValueError, KeyError):
        earlier = []
    call = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "run": a.run, "options": vars(a), "steps": []}

    def save():
        with open(path, "w") as f:
            json.dump({"calls": earlier + [call]}, f, indent=1)

    code, failed = 0, []
    for st in steps:
        print(f"== {st['name']} {time.strftime('%T')}", flush=True)
        rec = {"name": st["name"], "command": (["python", os.path.relpath(SUITE, ROOT), a.run] + st["argv"]) if st["argv"] else {"frames": st["frames"]},
               "fatal": st["fatal"], "tries": 0, "exit": None, "seconds": None}
        call["steps"].append(rec)
        save()                                                                      # the step that was running when a crash happened shows with exit null
        t0 = time.time()
        for _ in range(st["tries"]):
            rec["tries"] += 1
            rec["exit"] = runner(a, st)
            if rec["exit"] == 0:
                break
            print(f"{st['name']} failed (exit {rec['exit']}, try {rec['tries']} of {st['tries']})", flush=True)
            if st["store"]:
                set_aside(st["store"])
        rec["seconds"] = round(time.time() - t0, 1)
        save()
        if rec["exit"]:
            failed.append(st["name"])
            if st["fatal"]:
                code = 1
                break
    call["failed"], call["exit"] = failed, code
    save()
    print(f"== done {time.strftime('%T')}: " + ("all steps passed" if not failed else f"failed: {', '.join(failed)}" + ("" if code else " (not fatal)")) + f"; the call is in {path}", flush=True)
    return code


def main(argv=None) -> int:
    p = build_parser()
    a = p.parse_args(argv)
    if not (a.run and (a.quant or a.qual or a.videos or a.time or a.all)):
        p.print_help()
        return 2
    a.run = a.run.rstrip("/")
    if not os.path.isfile(os.path.join(a.run, "config.json")):
        p.error(f"{a.run} is not a run folder (no config.json); run from the bev_fusion folder")
    return execute(a, plan(a))


if __name__ == "__main__":
    raise SystemExit(main())
