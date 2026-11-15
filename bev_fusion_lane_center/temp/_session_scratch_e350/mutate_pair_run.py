"""mutation check of the pair-run tests (E350 phase A): each mutant = one reverted rule in a COPY of the source text of jwire_data.py / jwire_train.py (exec'd as a module; the files on disk are not touched);
the test of the module must fail on it. `HEAD` = the committed jwire_train.py (E325's code) with the new test function exec'd into it.
usage: bash tools/env/_py14.sh temp/_session_scratch_e350/mutate_pair_run.py [name ...]"""
import os
import re
import subprocess
import sys
import traceback
import types

R = os.environ["BFLC_ROOT"]
PD, PT = R + "/evaluation/jwire_data.py", R + "/training/train/jwire_train.py"
SRC = {PD: open(PD, encoding="utf-8").read(), PT: open(PT, encoding="utf-8").read()}

MUT = [  # (name, file, old, new, which test)
    ("M1 val_records gets the hard-coded E312 run", PT, "JD.val_records(seed, pair_run))", 'JD.val_records(seed, f"e312_p_s{seed}"))', "T"),
    ("M2 live_records without pair_run", PT, "workers, pair_run=pair_run))", "workers))", "T"),
    ("M3 VAL cache name without the pair run", PT, 'JD.cache_path("val", seed, pair_run=pair_run)', 'JD.cache_path("val", seed)', "T"),
    ("M4 TRAIN cache name without the pair run", PT, 'JD.cache_path("train", seed, max_train, pair_run=pair_run)', 'JD.cache_path("train", seed, max_train)', "T"),
    ("M5 checkpoint args without the pair run", PT, "args=dict(seed=seed, pair_run=pair_run, head_seed=h", "args=dict(seed=seed, head_seed=h", "T"),
    ("M6 wire_agreement.json without the pair run", PT, "info = dict(seed=seed, pair_run=pair_run, train_frames", "info = dict(seed=seed, train_frames", "T"),
    ("M7 existing run folder not refused", PT, 'assert not os.path.exists(os.path.join(ROOT, "runs", name, f)), f"runs/{name}/{f} exists: a finished run is never overwritten (give a new --name)"', "pass", "T"),
    ("M8 --name not mandatory with --pair-run", PT, "    assert name, f\"--name is mandatory", "    name = name or f\"e325_w_s{seed}\"\n    assert name, f\"--name is mandatory", "T"),
    ("M9 cv reads E325's VAL cache", PT, 'JD.cache_path("val", a.seed, pair_run=a.pair_run), a.seed, a.pair_run)', 'JD.cache_path("val", a.seed), a.seed, a.pair_run)', "T"),
    ("M10 the pair run is not passed from the CLI", PT, "pair_run=a.pair_run)\n        return", "pair_run=None)\n        return", "T"),
    ("M11 default run name changed", PT, 'return name or f"e325_w_s{seed}", JD.default_pair_run(seed)', 'return name or f"e350_w_s{seed}", JD.default_pair_run(seed)', "T"),
    ("M12 TRAIN cache loaded without the meta check", PD, 'assert obj["meta"] == want, f"{path}: meta', 'assert True or obj["meta"] == want, f"{path}: meta', "T"),
    ("M13 cache_path ignores the pair run", PD, "{'_pair_' + pair_run if other else ''}", "''", "D"),
    ("M14 meta not checked on load", PD, 'assert obj["meta"] == want, f"{path}: meta', 'assert True or obj["meta"] == want, f"{path}: meta', "D"),
    ("M15 a file without meta valid for any pair run", PD, 'assert want["pair_run"] == default_pair_run(seed), f"{path} has no meta', 'assert True or want["pair_run"] == default_pair_run(seed), f"{path} has no meta', "D"),
    ("M16 a cache can be overwritten", PD, 'assert not os.path.exists(path), f"{path} exists: a cache is never overwritten"', "pass", "D"),
    ("M17 val_records without the seed check", PD, "    check_seed(run_args(run), seed, run)\n", "", "D"),
    ("M18 check_seed ignores the seed in args", PD, 'assert args.get("seed", seed) == seed and (m is None', 'assert (m is None', "D"),
    ("M19 load_models takes E312's run", PD, "JE.load_model(pair_run, dev)", "JE.load_model(default_pair_run(seed), dev)", "D"),
    ("M20 live_records ignores its pair_run", PD, "load_models(seed, pair_run or default_pair_run(seed), dev)", "load_models(seed, default_pair_run(seed), dev)", "D"),
    ("M21 live pass without the seed check", PD, "    check_seed(args, seed, pair_run)\n    return mb, mp", "    return mb, mp", "D"),
    ("M22 val CLI: --run ignored", PD, "pr = a.pair_run or a.run", "pr = a.pair_run", "D"),
    ("M23 val CLI: --run and --pair-run may differ", PD, "assert a.run is None or a.pair_run is None or a.run == a.pair_run", "assert True", "D"),
    ("M24 val CLI writes E325's cache name", PD, 'path = cache_path("val", a.seed, a.max, a.thr, pr)', 'path = cache_path("val", a.seed, a.max, a.thr)', "D"),
    ("M25 live CLI without the pair run", PD, "device=a.device, pair_run=pr))", "device=a.device))", "D"),
    ("M26 default pair run named differently", PD, 'return f"e312_p_s{seed}"', 'return f"e312_p_s{seed}x"', "D"),
]


def load(name, path, text):
    mod = types.ModuleType(name)
    mod.__file__ = path
    mod.__dict__["__name__"] = name
    sys.modules[name] = mod
    exec(compile(text, path, "exec"), mod.__dict__)
    return mod


def run(mutant):
    name, path, old, new, which = mutant
    texts = dict(SRC)
    if old is not None:
        assert texts[path].count(old) == 1, f"{name}: the text to mutate occurs {texts[path].count(old)} times"
        texts[path] = texts[path].replace(old, new)
    JD = load("jwire_data", PD, texts[PD])
    JT = load("jwire_train", PT, texts[PT])
    (JD._test_pair_run if which == "D" else JT._test_pair_run)()


def head():
    """E325's committed jwire_train.py with the new test function: the test must fail on it (no pair_run anywhere)"""
    old = subprocess.run(["git", "show", "HEAD:bev_fusion_lane_center/training/train/jwire_train.py"], capture_output=True, text=True, cwd=R, check=True).stdout
    JD = load("jwire_data", PD, SRC[PD])
    JT = load("jwire_train", PT, old)
    m = re.search(r"\ndef _test_pair_run\(\):.*?(?=\n\n\ndef main)", SRC[PT], re.S)
    exec(compile(m.group(0), PT, "exec"), JT.__dict__)
    JT._test_pair_run()


if __name__ == "__main__":
    want = sys.argv[1:]
    todo = [("HEAD (E325's committed jwire_train.py)", None)] + [(m[0], m) for m in MUT]
    res = []
    for name, m in todo:
        if want and not any(name.startswith(w) for w in want):
            continue
        try:
            head() if m is None else run(m)
            res.append((name, "SURVIVED (test passes: a gap)", ""))
        except BaseException as e:
            tb = traceback.extract_tb(e.__traceback__)[-1]
            res.append((name, "killed", f"{type(e).__name__}: {str(e)[:110]}"))
        sys.stdout.flush()
    print("\n=== mutation result ===")
    for n, s, d in res:
        print(f"{s:30s} {n:55s} {d}")
    print(f"killed {sum(s == 'killed' for _, s, _ in res)} / {len(res)}")
