"""jarm.py - one arm of a token-head experiment in ONE command (train -> dumps -> TRAIN sweep -> pinned cell -> scores), and the table of arms. Replaces the per-experiment bash chains (sup / read / texts / choose scripts).

    wsl python3 tools/jarm.py run E383 f_s0 --seed 0 --head-seed 2 [--ref e373_cb_s0] [--trainer jtoken_fast_train.py] [--window] [--dry] -- --none-weight 0.25 [more trainer args]
    wsl python3 tools/jarm.py table e373_cb_s0 e383_f_s0 e383_k_s0 [--cell 0.3,0.2,0.6,off | --cell pinned]      # pooled VAL numbers per head, differences to the FIRST (the reference)
    wsl python3 tools/jarm.py selftest

`run` works on the run `<id lower>_<arm>` (here e383_f_s0); each step runs only when its output is missing (the same command continues a crashed or stopped arm) and the arm STOPS at the first step that fails:
  1. train       `<trainer> --name <run> --seed S --head-seed H --ident e292_a --steps 12000 --bs 8 --lr 1e-3 --workers 4 <the arguments after -->` (defaults = the champion recipe's frame; the extras are the arm's own,
                 e.g. `--none-weight 0.25 --cache runs/e383_cache --K 384`), relaunched (resume) up to 6 times; a `non-finite` line in the trainer's log (the card's kill) or a refusal of run.py (exit 2 / 3) stops it;
  2. dump_val, dump_train   the head on the cache tokens, VAL and the TRAIN sample (every 3), GPU; a dump that exists but is cut (a pickle without its end) stops the arm: it is never overwritten or deleted here;
  3. sweep       the per-class decode grid on the TRAIN sample (CPU);     4. choose   `jtoken_eval.py choose-cell` (E375's rule, written BEFORE any VAL score; no fallback: a missing cell stops the arm);
  5. scores      VAL at the pinned cell and at D115 (0.3, 0.2, 0.6, gate off); with --ref only the paired cards (`--ref-run <ref> --ref-run-cell D115`, same tokens only: NOT for a K-384 arm against K-512 heads).
Every step is launched by `run.py` with the experiment card's own words (title, prediction, rule and kill from experiments/<id>_*/experiment.md): no text to copy; ids `<id>.<step>_<arm>`; GPU steps ask for 6000 MB of RAM.
`--window` waits for the allowed GPU window (weekdays 08:30 - 17:30, nights 22:00 - 07:00) before every GPU launch. TEST is never touched (`--split val`). The numbers are the cards' (`runs/<run>/token_card_*.json`).
**What it does not do:** it decides nothing (verdicts stay by hand, by the card's rule); it does not enforce the card's kill beyond a non-finite loss (a loss-curve band or a no-learning rule is the caller's check); the E376 anatomy and ensembles are not steps.
"""
import argparse
import datetime
import glob
import json
import os
import subprocess
import sys
import time

ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
AR = os.path.join(os.path.dirname(ROOT), "autoresearch", "tools")
RUNS = os.path.join(ROOT, "runs")
D115 = "0.3,0.2,0.6,off"
RECORD = (0.3, 0.3, 0.5, True)                                                          # the decode of record: link 0.3 (long = short), keep 0.5, gate on: its cards have no suffix
METRICS = ("topo_F1", "F1@1.0", "lanes_found", "junction_F1", "dup_frac")
DEFAULTS = "--ident e292_a --steps 12000 --bs 8 --lr 1e-3 --workers 4"
PYR, PY = "bash tools/env/_pyr.sh", "env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 10 bash tools/env/_py14.sh"


def card_texts(eid, root=ROOT):
    """(hypothesis, prediction, rule) of the experiment card of `eid` (experiments/<eid>_*/experiment.md, its json block): the title, the prediction and the rule clauses with the kill text; no card = SystemExit (a run needs its card)"""
    sys.path.insert(0, AR)
    import experiment as X
    dirs = sorted(glob.glob(os.path.join(root, "experiments", f"{eid}_*")))
    if not dirs:
        raise SystemExit(f"no experiment folder experiments/{eid}_*: write the card first")
    card = X.read(X.source(dirs[0]))
    pr = card.get("prediction")
    pred = pr if isinstance(pr, str) else f"{pr.get('value')} vs {pr.get('vs')}" if isinstance(pr, dict) else str(pr)
    return (card.get("title", eid), pred, "the card's rule (hard rule 4: written before the run): " + json.dumps(card.get("rule", []), separators=(",", ":")) + " | kill: " + str(card.get("kill", "")))


def in_window(now=None):
    """the allowed GPU windows (the lead's rule): weekdays 08:30 - 17:30, nights 22:00 - 07:00"""
    now = now or datetime.datetime.now()
    hm = now.hour * 100 + now.minute
    return hm >= 2200 or hm < 700 or (now.weekday() < 5 and 830 <= hm < 1730)


def parse_cell(cell):
    s, l, k, g = cell.split(",")
    return float(s), float(l), float(k), g == "on"


def cell_args(cell):
    """'0.3,0.2,0.6,off' -> the decode arguments of `jtoken_eval.py score`"""
    s, l, k, g = cell.split(",")
    return f"--link-thr {s} --link-thr-long {l} --keep-thr {k}" + (" --no-none-gate" if g == "off" else "")


def card_path(run, cell, ref=None, ref_cell=D115):
    """the score card a (run, cell[, ref]) writes, by the evaluator's own naming (`jtoken_eval.variant_suffix` / `cell_sfx`): token_card_rec[_lt<S>[_ltl<L> when long differs]_kt<K>[_nng]][_vs_<ref>[_rc<cell> when the
    reference is decoded at another cell than this one]].json"""
    s, l, k, g = parse_cell(cell)
    suffix = "" if (s, l, k, g) == RECORD else f"_lt{s:g}" + ("" if l == s else f"_ltl{l:g}") + f"_kt{k:g}" + ("_nng" if not g else "")
    name = "token_card_rec" + suffix
    if ref:
        rc = parse_cell(ref_cell)
        name += f"_vs_{ref}" + ("" if rc == (s, l, k, g) else f"_rc{rc[0]:g}_{rc[1]:g}_{rc[2]:g}_{'on' if rc[3] else 'off'}")
    return os.path.join(RUNS, run, name + ".json")


def metrics(run, cell, ref=None):
    """the pooled VAL numbers of a head at a cell, from its score card ({} when the card is missing; a paired card of the same cell holds the same numbers); a card of another split is refused"""
    p = card_path(run, cell, ref)
    if not os.path.exists(p):
        alt = sorted(glob.glob(p[:-5] + "_vs_*.json")) if ref is None else []
        if not alt:
            return {}
        p = alt[0]
    c = json.load(open(p))
    assert c.get("split", "val") == "val", f"{p}: a {c.get('split')} card in a VAL table"
    return {**{k: c["metrics"][k] for k in METRICS if k in c["metrics"]}, "_frames": c.get("frames")}


def pinned_cell(run):
    """the cell `choose-cell` pinned for the run (runs/<run>/token_cell_perclass*.json, 'cell' = 'S,L,K,on|off'), else None"""
    fs = sorted(f for f in glob.glob(os.path.join(RUNS, run, "token_cell_perclass*.json")) if "_mean_" not in f and "_ref" not in f)
    return json.load(open(fs[-1])).get("cell") if fs else None


def dump_cut(path):
    """True when a dump pickle exists but does not end with the pickle STOP opcode (a write that was cut: the evaluator writes it in place)"""
    if not os.path.exists(path):
        return False
    if os.path.getsize(path) == 0:
        return True
    with open(path, "rb") as fh:
        fh.seek(-1, 2)
        return fh.read(1) != b"."


class Arm:
    def __init__(self, eid, arm, seed, head_seed, ref=None, trainer="jtoken_train.py", extra="", window=False, dry=False, texts=None):
        self.eid, self.arm, self.seed, self.head_seed, self.ref, self.trainer, self.extra, self.window, self.dry = eid, arm, seed, head_seed, ref, trainer, extra, window, dry
        assert not arm.startswith(("vs_", "cell", "d115", "dump", "sweep", "choose")), f"arm name {arm}: it would make run ids of another arm's steps"
        self.run = f"{eid.lower()}_{arm}"
        self.texts = texts or (("dry", "dry", "dry") if dry else card_texts(eid))

    def say(self, msg):
        print(f"{time.strftime('%m-%d %H:%M:%S')} [jarm {self.run}] {msg}", flush=True)

    def launch(self, step, cmd, gpu, timeout, retries=0):
        """one run.py launch -> its exit code (0 = the run finished)"""
        while gpu and self.window and not in_window():
            time.sleep(60)
        h, p, r = self.texts
        rid = f"{self.eid}.{step}_{self.arm}" if step else f"{self.eid}.{self.arm}"
        line = ["python3", os.path.join(AR, "run.py"), "--id", rid, "--seed", str(self.seed), "--retries", str(retries), "--require-card", "--split", "val", "--timeout", str(timeout), "--hyp", h, "--pred", p, "--rule", r]
        line += (["--gpu", "--ram-mb", "6000"] if gpu else []) + ["--", "bash", "-c", cmd]
        self.say(f"{step or 'train'}: {cmd[:150]}")
        return 0 if self.dry else subprocess.call(line, cwd=ROOT, env=dict(os.environ, BFLC_SIG="1.0"))

    def out(self, name):
        return os.path.join(RUNS, self.run, name)

    def step(self, name, cmd, outputs, gpu, timeout, retries=0):
        """run `cmd` unless all `outputs` exist; True when they exist afterwards (a failed step or a missing output stops the arm)"""
        if not self.dry and all(glob.glob(o) for o in outputs):
            return True
        rc = self.launch(name, cmd, gpu, timeout, retries)
        ok = self.dry or (rc == 0 and all(glob.glob(o) for o in outputs))
        if not ok:
            self.say(f"STOPPED: step {name} exit {rc}" + ("" if rc else f", output missing: {outputs}"))
        return ok

    def done_line(self):
        log = self.out("log.txt")
        return os.path.exists(log) and any(l.split()[-2:] == ["done", self.run] for l in open(log, encoding="utf-8", errors="replace") if l.strip())

    def train(self):
        log = self.out("log.txt")
        for _ in range(6):
            if self.done_line():
                return True
            if os.path.exists(log) and "non-finite" in open(log, encoding="utf-8", errors="replace").read():      # the trainer's line `step N: non-finite <loss|gradient> skipped`
                self.say("STOPPED: a non-finite loss or gradient in the log (the card's kill)")
                return False
            rc = self.launch("", f"{PYR} {self.trainer} --name {self.run} --seed {self.seed} --head-seed {self.head_seed} {DEFAULTS} {self.extra}", True, 90)
            if self.dry:
                return True
            if rc in (2, 3):                                                       # run.py refused the launch (a card, a hold, a slot rule) or it is blocked: a relaunch would be refused again
                self.say(f"STOPPED: run.py exit {rc} (refused or blocked)")
                return False
        if not self.done_line():
            self.say("STOPPED: 6 launches and the run is not done")
        return self.done_line()

    def readout(self):
        """-> the pinned cell, or None when a step failed (nothing is scored at a cell that was not chosen on TRAIN)"""
        S, R = self.seed, self.run
        for f in ("token_val.pkl", "token_train_every3.pkl"):
            if dump_cut(self.out(f)):
                self.say(f"STOPPED: {self.out(f)} is cut (a write that did not finish); it is never overwritten: move it away by hand")
                return None
        for name, cmd, outs, gpu, to, rt in (
                ("dump_val", f"{PYR} jtoken_eval.py dump --run {R} --seed {S} --split val", [self.out("token_val.pkl")], True, 60, 1),
                ("dump_train", f"{PYR} jtoken_eval.py dump --run {R} --seed {S} --split train --every 3", [self.out("token_train_every3.pkl")], True, 60, 1),
                ("sweep", f"{PY} jtoken_eval.py sweep --run {R} --seed {S} --split train --every 3 --grid perclass", [self.out("token_sweep_perclass_train_every3.json")], False, 120, 0),
                ("choose", f"{PY} jtoken_eval.py choose-cell --runs {R} --ref-cell {D115} --split train --every 3", [self.out("token_cell_perclass_train_every3.json")], False, 20, 0)):
            if not self.step(name, cmd, outs, gpu, to, rt):
                return None
        cell = pinned_cell(R) if not self.dry else D115
        if not cell:
            self.say("STOPPED: no pinned cell")
            return None
        self.say(f"pinned cell {cell} (chosen on the TRAIN sample, before any VAL score)")
        for c in dict.fromkeys([cell, D115]):
            rc = f" --ref-run {self.ref} --ref-run-cell {D115}" if self.ref else ""
            if not self.step(("d115" if c == D115 else "cell") + ("_vs" if self.ref else ""), f"{PY} jtoken_eval.py score --run {R} --seed {S} {cell_args(c)}{rc}", [card_path(R, c, self.ref)], False, 60):
                return None
        return cell

    def go(self):
        if not self.train():
            return 1
        if self.readout() is None:
            return 1
        table([self.ref, self.run] if self.ref else [self.run], "pinned")
        return 0


def table(runs, cell):
    """print the pooled VAL numbers of the runs at a cell ('pinned' = each run's own pinned cell, D115 when it has none), the differences to the first run (dup_frac: lower is better; the paired intervals are in the cards)"""
    rows = [(r, (pinned_cell(r) or D115) if cell == "pinned" else cell) for r in runs]
    rows = [(r, c, metrics(r, c)) for r, c in rows]
    print("VAL, pooled over the frames of the card (" + ", ".join(sorted({str(m.get("_frames")) for _, _, m in rows if m})) + " frames); dup_frac: lower is better")
    print("run".ljust(18) + "cell".ljust(16) + "".join(m.rjust(12) for m in METRICS))
    for r, c, m in rows:
        print(r.ljust(18) + c.ljust(16) + "".join((f"{m[k]:.4f}" if k in m else "-").rjust(12) for k in METRICS))
    base = rows[0][2]
    for r, c, m in rows[1:]:
        print(("  " + r + " - " + rows[0][0]).ljust(34) + "".join((f"{m[k] - base[k]:+.4f}" if k in m and k in base else "-").rjust(12) for k in METRICS))


class FakeArm(Arm):
    """for the selftest: `launch` records the step and creates its outputs unless the step is listed in `fail`"""

    def __init__(self, *a, fail=(), **k):
        super().__init__(*a, texts=("h", "p", "r"), **k)
        self.fail, self.log = set(fail), []
        os.makedirs(os.path.join(RUNS, self.run), exist_ok=True)

    def launch(self, step, cmd, gpu, timeout, retries=0):
        self.log.append(step)
        if step in self.fail:
            return 1
        d = os.path.join(RUNS, self.run)
        if step == "":
            open(os.path.join(d, "log.txt"), "a").write(f"05:00:00 done {self.run}\n")
        elif step == "dump_val":
            open(os.path.join(d, "token_val.pkl"), "wb").write(b"x.")
        elif step == "dump_train":
            open(os.path.join(d, "token_train_every3.pkl"), "wb").write(b"x.")
        elif step == "sweep":
            open(os.path.join(d, "token_sweep_perclass_train_every3.json"), "w").write("{}")
        elif step == "choose":
            json.dump(dict(cell="0.2,0.2,0.6,off"), open(os.path.join(d, "token_cell_perclass_train_every3.json"), "w"))
        else:
            ref = self.ref if step.endswith("_vs") else None
            c = D115 if step.startswith("d115") else "0.2,0.2,0.6,off"
            json.dump(dict(split="val", frames=873, metrics={k: 0.5 for k in METRICS}), open(card_path(self.run, c, ref), "w"))
        return 0


def selftest():
    import tempfile
    global RUNS
    a = parse(["run", "E383", "f_s0", "--seed", "0", "--head-seed", "2", "--ref", "r", "--", "--none-weight", "0.25"])
    assert a.cmd == "run" and a.extra == ["--none-weight", "0.25"] and a.id == "E383" and a.arm == "f_s0" and a.ref == "r" and a.seed == 0 and a.head_seed == 2
    assert cell_args(D115) == "--link-thr 0.3 --link-thr-long 0.2 --keep-thr 0.6 --no-none-gate" and cell_args("0.2,0.1,0.5,on") == "--link-thr 0.2 --link-thr-long 0.1 --keep-thr 0.5"
    n = lambda *x: os.path.basename(card_path(*x))  # noqa: E731
    assert n("r", D115) == "token_card_rec_lt0.3_ltl0.2_kt0.6_nng.json"                         # the real names of runs/e373_cb_s0, e376_mx_s0 (evaluator's naming)
    assert n("r", "0.2,0.2,0.6,off") == "token_card_rec_lt0.2_kt0.6_nng.json", "no _ltl when long == short"
    assert n("r", "0.3,0.3,0.5,on") == "token_card_rec.json", "the decode of record has no suffix"
    assert n("r", D115, "q") == "token_card_rec_lt0.3_ltl0.2_kt0.6_nng_vs_q.json"
    assert n("r", "0.2,0.2,0.6,off", "q") == "token_card_rec_lt0.2_kt0.6_nng_vs_q_rc0.3_0.2_0.6_off.json", "the reference decoded at another cell"
    mon, sat = datetime.datetime(2026, 11, 16, 12, 0), datetime.datetime(2026, 11, 14, 12, 0)
    assert in_window(mon) and not in_window(sat) and in_window(datetime.datetime(2026, 11, 14, 23, 0)) and not in_window(datetime.datetime(2026, 11, 16, 18, 0)) and not in_window(datetime.datetime(2026, 11, 16, 8, 0)) and in_window(datetime.datetime(2026, 11, 17, 6, 59))
    with tempfile.TemporaryDirectory() as d:
        keep, RUNS = RUNS, d
        try:
            for name, v in (("ref", 0.67), ("x", 0.66)):
                os.makedirs(os.path.join(d, name))
                json.dump(dict(split="val", frames=873, metrics={k: v for k in METRICS}), open(card_path(name, D115), "w"))
            assert metrics("x", D115)["topo_F1"] == 0.66 and metrics("nope", D115) == {}
            json.dump(dict(cell=D115), open(os.path.join(d, "x", "token_cell_perclass_train_every3.json"), "w"))
            assert pinned_cell("x") == D115 and pinned_cell("ref") is None
            json.dump(dict(split="train", frames=1, metrics={}), open(card_path("ref", "0.1,0.1,0.5,on"), "w"))
            try:
                metrics("ref", "0.1,0.1,0.5,on")
                raise SystemExit("a train card in a VAL table must stop")
            except AssertionError:
                pass
            import contextlib
            import io
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                table(["ref", "x"], D115)
            assert "-0.0100" in buf.getvalue() and "x - ref" in buf.getvalue() and "873 frames" in buf.getvalue(), buf.getvalue()
            os.makedirs(os.path.join(d, "y"))                                           # a run with only a paired card of the same decode: the numbers come from it
            json.dump(dict(split="val", frames=873, metrics={k: 0.7 for k in METRICS}), open(card_path("y", D115, "ref"), "w"))
            assert metrics("y", D115)["topo_F1"] == 0.7
            assert os.path.basename(card_path("y", "0.3,0.3,0.5,on", "ref")) == "token_card_rec_vs_ref_rc0.3_0.2_0.6_off.json"
            p = os.path.join(d, "x", "token_val.pkl")
            open(p, "wb").write(b"abc")
            assert dump_cut(p) and not dump_cut(os.path.join(d, "x", "none.pkl"))
            open(p, "wb").write(b"abc.")
            assert not dump_cut(p)
            open(p, "wb").write(b"")
            assert dump_cut(p), "an empty dump is cut"
            exp = os.path.join(d, "experiments", "E9_t")
            os.makedirs(exp)
            open(os.path.join(exp, "experiment.md"), "w").write('# E9\n```json card\n{"id":"E9","title":"a title","prediction":{"value":0.1,"vs":"ref"},"rule":[{"metric":"Q"}],"kill":"k","arms":{"a":{"run_ids":["E9.a"]}}}\n```\n')
            t = card_texts("E9", d)
            assert t[0] == "a title" and t[1] == "0.1 vs ref" and '"metric":"Q"' in t[2] and t[2].endswith("| kill: k")
            try:
                card_texts("E8", d)
                raise AssertionError("no folder must stop")
            except SystemExit:
                pass
            g = FakeArm("E9", "a", 0, 2, ref="ref")                                       # a full arm: every step in order; the pinned cell is not D115, so both are scored, paired
            assert g.go() == 0 and g.log == ["", "dump_val", "dump_train", "sweep", "choose", "cell_vs", "d115_vs"], g.log
            assert os.path.exists(card_path("e9_a", "0.2,0.2,0.6,off", "ref")) and os.path.exists(card_path("e9_a", D115, "ref"))
            g2 = FakeArm("E9", "a", 0, 2, ref="ref")
            assert g2.go() == 0 and g2.log == [], "everything exists: nothing is launched again"
            h = FakeArm("E9", "b", 0, 2, fail=("sweep",))                               # a failed sweep stops the arm: no cell, no score
            assert h.go() == 1 and h.log == ["", "dump_val", "dump_train", "sweep"] and not glob.glob(os.path.join(d, "e9_b", "token_cell*")), h.log
            c = FakeArm("E9", "c", 0, 2)                                                  # a cut dump stops it before any read-out step
            open(os.path.join(d, "e9_c", "log.txt"), "w").write("05:00:00 done e9_c\n")
            open(os.path.join(d, "e9_c", "token_val.pkl"), "wb").write(b"cut")
            assert c.go() == 1 and c.log == [], c.log
            for text, want in (("12:00:00 done e9_d\n", True), ("12:00:00 step 5: non-finite loss skipped (1 so far, 1 in a row)\n", False)):  # the training log: done / the trainer's own non-finite line
                f = FakeArm("E9", "d", 0, 2)
                open(os.path.join(d, "e9_d", "log.txt"), "w").write(text)
                assert f.train() == want and f.log == [], (text, f.log)
            open(os.path.join(d, "e9_d", "log.txt"), "w").write("12:00:00 step 1 loss 3.2\n")  # neither: relaunched 6 times, then stopped
            f = FakeArm("E9", "d", 0, 2, fail=("",))
            assert f.train() is False and f.log == [""] * 6, f.log
            f = FakeArm("E9", "d", 0, 2)                                                  # run.py exit 2 (refused): no relaunch
            f.launch = lambda step, *a, **k: (f.log.append(step), 2)[1]
            assert f.train() is False and f.log == [""], f.log
            e = FakeArm("E9", "e", 0, 2)                                                  # rc 0 but the output is missing: the arm stops
            e.launch = lambda step, *a, **k: (e.log.append(step), 0)[1]
            open(os.path.join(d, "e9_e", "log.txt"), "w").write("05:00:00 done e9_e\n")
            assert e.go() == 1 and e.log == ["dump_val"], e.log
            try:
                FakeArm("E9", "vs_f", 0, 2)
                raise SystemExit("an arm name that collides with a step id must be refused")
            except AssertionError:
                pass
        finally:
            RUNS = keep
    print("jarm selftest ok")


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("id")
    r.add_argument("arm")
    r.add_argument("--seed", type=int, required=True)
    r.add_argument("--head-seed", type=int, required=True, dest="head_seed")
    r.add_argument("--ref")
    r.add_argument("--trainer", default="jtoken_train.py")
    r.add_argument("--window", action="store_true")
    r.add_argument("--dry", action="store_true")
    t = sub.add_parser("table")
    t.add_argument("runs", nargs="+")
    t.add_argument("--cell", default=D115)
    sub.add_parser("selftest")
    return ap


def parse(argv):
    """argv -> namespace; the words after a lone `--` are the trainer's own arguments (`extra`)"""
    argv = list(argv)
    extra = []
    if "--" in argv:
        i = argv.index("--")
        argv, extra = argv[:i], argv[i + 1:]
    a = build_parser().parse_args(argv)
    a.extra = extra
    return a


def main():
    a = parse(sys.argv[1:])
    if a.cmd == "selftest":
        return selftest()
    if a.cmd == "table":
        return table(a.runs, a.cell)
    sys.exit(Arm(a.id, a.arm, a.seed, a.head_seed, a.ref, a.trainer, " ".join(a.extra), a.window, a.dry).go())


if __name__ == "__main__":
    main()
