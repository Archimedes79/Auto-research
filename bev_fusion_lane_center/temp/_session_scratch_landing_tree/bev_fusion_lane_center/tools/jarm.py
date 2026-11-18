"""jarm.py - one arm of a token-head experiment in ONE command (train -> dumps -> TRAIN sweep -> pinned cell -> scores), and the table of arms. Replaces the per-experiment bash chains (sup / read / texts / choose scripts).

    wsl python3 tools/jarm.py run E383 f_s0 --seed 0 --head-seed 2 [--ref e373_cb_s0] [--trainer jtoken_fast_train.py] [--curve [STEP:DIFF,...]] [--window] [--dry] -- --none-weight 0.25 [more trainer args]
    wsl python3 tools/jarm.py ids E383 f_s0 f_s1 [--ref e373_cb_s0]                                                  # the run ids the arms launch, marked when the card's run_ids lack them (add them to the card)
    wsl python3 tools/jarm.py table e373_cb_s0 e383_f_s0 e383_k_s0 [--cell 0.3,0.2,0.6,off | --cell pinned]      # pooled VAL numbers per head, differences to the FIRST (the reference)
    wsl python3 tools/jarm.py curve e383_f_s0 e373_cb_s0 [--band 100:0.3,600:0.05]                                  # a speed-change arm vs its reference: the data-order fingerprint (gt_links / pos_pairs / gt_outside equal at every logged step) + the loss band
                                                                                                                    # (default 0.05 / 0.03 / 0.02 at steps 500 / 2000 / 12000: a gross-failure check, not twin evidence); exit 1 outside
    wsl python3 tools/jarm.py selftest

`run` works on the run `<id lower>_<arm>` (here e383_f_s0); the run ids come from the card: the training id `<E-id>.<arm>` must be listed in an arm's `run_ids` (else the run is refused here, before anything starts), the read-out ids `<E-id>.<step>_<arm>` are listed
by `jarm.py ids` (put them into the card to have its rule registered for every step; unlisted ones run as diagnostics and say so); each step runs only when its output is missing (the same command continues a crashed or stopped arm) and the arm STOPS at the first step that fails:
  1. train       `<trainer> --name <run> --seed S --head-seed H --ident e292_a --steps 12000 --bs 8 --lr 1e-3 --workers 4 <the arguments after -->` (defaults = the champion recipe's frame; the extras are the arm's own,
                 e.g. `--none-weight 0.25 --cache runs/e383_cache --K 384`), relaunched (resume) up to 6 times; a `non-finite` line in the trainer's log (the card's kill) or a refusal of run.py (exit 2 / 3) stops it;
  2. dump_val, dump_train   the head on the cache tokens, VAL and the TRAIN sample (every 3), GPU; a dump that exists but is cut (a pickle without its end) stops the arm: it is never overwritten or deleted here;
  3. sweep       the per-class decode grid on the TRAIN sample (CPU);     4. choose   `jtoken_eval.py choose-cell` (E375's rule, written BEFORE any VAL score; no fallback: a missing cell stops the arm);
  5. scores      VAL at the pinned cell and at D115 (0.3, 0.2, 0.6, gate off); with --ref only the paired cards (`--ref-run <ref> --ref-run-cell D115`, same tokens only: NOT for a K-384 arm against K-512 heads).
Every step is launched by `run.py` with the experiment card's own words (title, prediction, rule and kill from experiments/<id>_*/experiment.md): no text to copy; ids `<id>.<step>_<arm>`; GPU steps ask for 6000 MB of RAM.
One jarm per arm (a lock /tmp/jarm_<run>.lock; a second one says so and exits 0) and a launch waits while a `run.py --id <id> ` of another supervisor is alive: what `tools/maintenance/keep_alive.sh` did for the chain supervisors (log, lock, wait, relaunch until done), which `jarm` absorbs.
`--window` waits for the allowed GPU window (weekdays 08:30 - 17:30, nights 22:00 - 07:00) before every GPU launch. TEST is never touched (`--split val`). The numbers are the cards' (`runs/<run>/token_card_*.json`).
**What it does not do:** it decides nothing (verdicts stay by hand, by the card's rule); it does not enforce the card's kill beyond a non-finite loss and, with --curve (needs --ref), the `curve` check after training (a fired check stops the arm: NOT-PARITY; a no-learning rule is the caller's check); the E376 anatomy and ensembles are not steps.
"""
import argparse
import datetime
import fcntl
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


def card_run_ids(eid, root=ROOT):
    """every run id the experiment card of `eid` lists in its arms (`run_ids`): the ids whose launches get the card's rule registered (`experiment.card_for_run`, run.py: card_sha) and that `decide.py` reads"""
    sys.path.insert(0, AR)
    import experiment as X
    dirs = sorted(glob.glob(os.path.join(root, "experiments", f"{eid}_*")))
    card = X.read(X.source(dirs[0])) if dirs else {}
    return {r for a in (card.get("arms") or {}).values() if isinstance(a, dict) for r in (a.get("run_ids") or [])}


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
    def __init__(self, eid, arm, seed, head_seed, ref=None, trainer="jtoken_train.py", extra="", window=False, dry=False, texts=None, curve_check=None):
        self.eid, self.arm, self.seed, self.head_seed, self.ref, self.trainer, self.extra, self.window, self.dry = eid, arm, seed, head_seed, ref, trainer, extra, window, dry
        self.curve_check = curve_check
        assert not arm.startswith(("vs_", "cell", "d115", "dump", "sweep", "choose")), f"arm name {arm}: it would make run ids of another arm's steps"
        self.run = f"{eid.lower()}_{arm}"
        self.texts = texts or (("dry", "dry", "dry") if dry else card_texts(eid))

    def rid(self, step):
        """the run.py id of a step: `<E-id>.<arm>` for the training, `<E-id>.<step>_<arm>` for the others"""
        return f"{self.eid}.{step}_{self.arm}" if step else f"{self.eid}.{self.arm}"

    def step_ids(self):
        """every run id this arm launches, in order (the training, the read-out steps; the cell steps carry `_vs` with --ref): the list a card names in `run_ids` to have its rule registered for all of them"""
        return [self.rid(x) for x in ("", "dump_val", "dump_train", "sweep", "choose", "cell_vs" if self.ref else "cell", "d115_vs" if self.ref else "d115")]

    def check_ids(self, listed):
        """the run ids come from the card: the training id MUST be listed (else run.py registers no rule for the arm, and `decide.py` has no run to gate) -> SystemExit; read-out ids that are not listed run as diagnostics
        without a registered rule: say so once, with the line to add. `listed` = `card_run_ids`"""
        if self.rid("") not in listed:
            raise SystemExit(f"the card of {self.eid} does not list the run id {self.rid('')} in the run_ids of an arm: add it (and the read-out ids: `jarm.py ids {self.eid} {self.arm}{' --ref ' + self.ref if self.ref else ''}`) before the run: a run without its card is refused (hard rule 4)")
        miss = [x for x in self.step_ids() if x not in listed]
        if miss:
            self.say(f"note: {len(miss)} read-out run id(s) are not listed in the card's run_ids ({', '.join(miss[:3])}{' ...' if len(miss) > 3 else ''}): they run as diagnostics without the card's rule registered; list them to register it")

    def say(self, msg):
        print(f"{time.strftime('%m-%d %H:%M:%S')} [jarm {self.run}] {msg}", flush=True)

    def alive(self, rid):
        """True while a `run.py --id <rid> ` of another supervisor is alive (a second launch of the same id would collide): the shared part of the old keep_alive.sh"""
        out = subprocess.run(["pgrep", "-f", f"run.py --id {rid} "], capture_output=True, text=True).stdout.split()
        return any(int(x) != os.getpid() for x in out)

    def lock(self):
        """one jarm per arm: an exclusive lock /tmp/jarm_<run>.lock held until the process ends -> True; False = another jarm runs this arm (say so and stop with exit 0)"""
        self._lock = open(f"/tmp/jarm_{self.run}.lock", "w")
        try:
            fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return False
        return True

    def launch(self, step, cmd, gpu, timeout, retries=0):
        """one run.py launch -> its exit code (0 = the run finished)"""
        while gpu and self.window and not in_window():
            time.sleep(60)
        while not self.dry and self.alive(self.rid(step)):
            self.say(f"{self.rid(step)} is alive under another supervisor: waiting")
            time.sleep(30)
        h, p, r = self.texts
        rid = self.rid(step)
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
        if self.curve_check is not None and self.ref and not self.dry:                              # a speed-change arm: a fired check = NOT-PARITY, chase before use (no dumps, no scores)
            ok, text = curve(self.run, self.ref, parse_band(self.curve_check))
            self.say(("curve check ok: " if ok else "STOPPED: curve check fired (NOT-PARITY, chase the difference before anything uses this path): ") + text)
            if not ok:
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


BAND = {500: 0.05, 2000: 0.03, 12000: 0.02}
def parse_band(s):
    """'STEP:DIFF,STEP:DIFF' -> {step: diff}; None or '' = the default `BAND`"""
    return BAND if not s else {int(k): float(d) for k, d in (x.split(":") for x in s.split(","))}


FINGERPRINT = ("gt_links", "pos_pairs", "gt_outside")                                    # integer columns of the log: they come from the batch alone, so equal columns = the same data order


def curve(run, ref, band=BAND):
    """the gross-failure check of a speed-change arm against its reference (a loose band, NOT twin evidence: head seeds of the same recipe differ by a third of it): (1) the data-order fingerprint, the integer columns `FINGERPRINT` of runs/<run>/log.jsonl
    must equal the reference's at EVERY logged step both runs have (and there must be one: the GPU loader reproduces the plain loader's order); (2) the logged `total` against the same step of runs/<ref> at the steps of `band` -> (ok, text);
    a missing file or step is not ok"""
    def load(r):
        f = os.path.join(RUNS, r, "log.jsonl")
        return {j["step"]: j for j in (json.loads(l) for l in open(f)) if "total" in j} if os.path.exists(f) else {}
    a, b = load(run), load(ref)
    rows = {s: (a.get(s, {}).get("total"), b.get(s, {}).get("total")) for s in band}
    ok_band = all(x is not None and y is not None and abs(x - y) <= band[s] for s, (x, y) in rows.items())
    common = sorted(set(a) & set(b))
    bad = [s for s in common if any(a[s].get(k) != b[s].get(k) for k in FINGERPRINT)]
    ok_order = bool(common) and not bad
    text = f"{run} vs {ref}: " + ", ".join(f"step {s}: " + ("missing" if x is None or y is None else f"{x:.4f} vs {y:.4f} (diff {x - y:+.4f}, band {band[s]})") for s, (x, y) in rows.items())
    text += f"; data order {'equal at all ' + str(len(common)) + ' common logged steps' if ok_order else ('differs at steps ' + str(bad[:5]) if bad else 'no common logged step')}"
    return ok_band and ok_order, text + (" -> inside the band, same order" if ok_band and ok_order else " -> OUTSIDE the band, other order or missing")


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
            base = {500: 0.70, 2000: 0.48, 12000: 0.36}
            for r, rows, gl in (("cA", base, 100), ("cB", {500: 0.72, 2000: 0.49, 12000: 0.37}, 100), ("cC", {**base, 12000: 0.40}, 100), ("cD", {**base, 500: 0.76}, 100), ("cE", {**base, 2000: 0.52}, 100),
                                ("cF", {**base, 12000: 0.33}, 100), ("cG", base, 101)):          # cG: another data order (gt_links of the batch differs)
                os.makedirs(os.path.join(d, r))
                open(os.path.join(d, r, "log.jsonl"), "w").write("".join(json.dumps(dict(step=s, total=v, gt_links=gl + s, pos_pairs=gl, gt_outside=3)) + chr(10) for s, v in rows.items()) + json.dumps(dict(step=7, val={})) + chr(10))
            assert curve("cB", "cA")[0] and curve("cB", "cA", {500: 0.05})[0], "inside the band"
            assert not curve("cC", "cA")[0] and not curve("cD", "cA")[0] and not curve("cE", "cA")[0] and not curve("cF", "cA")[0], "outside at 12000 (+), at 500, at 2000, at 12000 (-): each step and both signs count"
            assert not curve("nope", "cA")[0] and not curve("cA", "nope")[0], "a missing run is not ok"
            ok_g, text_g = curve("cG", "cA")
            assert not ok_g and "differs at steps" in text_g, "the same losses on another data order are not parity (the fingerprint columns)"
            lk1, lk2 = FakeArm("E9", "lk", 0, 2), FakeArm("E9", "lk", 0, 2)                # one jarm per arm: the second lock is refused while the first is held, and free again after
            assert lk1.lock() and not lk2.lock(), "a second jarm of the same arm must not get the lock"
            lk1._lock.close()
            assert lk2.lock(), "the lock is free again when the holder ended"
            lk2._lock.close()
            al = FakeArm("E9", "al", 0, 2)
            assert not al.alive("E9.nothing_runs_under_this_id"), "no process: not alive"
            g0 = FakeArm("E9", "a", 0, 2, ref="ref")                                      # the run ids come from the card
            assert g0.step_ids() == ["E9.a", "E9.dump_val_a", "E9.dump_train_a", "E9.sweep_a", "E9.choose_a", "E9.cell_vs_a", "E9.d115_vs_a"], g0.step_ids()
            assert FakeArm("E9", "a", 0, 2).step_ids()[-2:] == ["E9.cell_a", "E9.d115_a"]
            g0.check_ids(set(g0.step_ids()))                                             # all listed: silent and fine
            g0.check_ids({"E9.a"})                                                       # only the training listed: a note, no stop
            try:
                g0.check_ids({"E9.b", "E9.dump_val_a"})
                raise AssertionError("an unlisted training id must stop the run")
            except SystemExit as e:
                assert "E9.a" in str(e) and "jarm.py ids" in str(e), e
            assert card_run_ids("E9", d) == {"E9.a"}, card_run_ids("E9", d)
            assert parse_band(None) == BAND and parse_band("100:0.15,500:0.05") == {100: 0.15, 500: 0.05}
            try:
                parse(["run", "E9", "x", "--seed", "0", "--head-seed", "2", "--curve"])
                raise AssertionError("--curve without --ref must stop")
            except SystemExit:
                pass
            for ref_, want, log_ in (("cA", 0, ["dump_val", "dump_train", "sweep", "choose", "cell_vs", "d115_vs"]), ("cG", 1, [])):          # the curve check after the training: ok -> the read-out runs, fired -> nothing runs
                k = FakeArm("E9", "k_" + ref_, 0, 2, ref=ref_)
                k.curve_check = ""
                os.makedirs(os.path.join(d, "e9_k_" + ref_), exist_ok=True)
                open(os.path.join(d, "e9_k_" + ref_, "log.txt"), "w").write("05:00:00 done e9_k_" + ref_ + chr(10))
                open(os.path.join(d, "e9_k_" + ref_, "log.jsonl"), "w").write("".join(json.dumps(dict(step=s_, total=v, gt_links=100 + s_, pos_pairs=100, gt_outside=3)) + chr(10) for s_, v in base.items()))
                assert k.go() == want and k.log[:6] == log_[:len(k.log[:6])] and (want == 1) == (k.log == []), (ref_, k.log)
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
    r.add_argument("--curve", nargs="?", const="", default=None, metavar="BAND", help="a speed-change arm: after training run `curve` against --ref (data-order fingerprint + loss band, optionally STEP:DIFF,... as `curve --band`); a fired check stops the arm before any read-out step")
    i = sub.add_parser("ids", help="the run ids an arm launches (for the card's run_ids)")
    i.add_argument("id")
    i.add_argument("arm", nargs="+")
    i.add_argument("--ref")
    t = sub.add_parser("table")
    t.add_argument("runs", nargs="+")
    t.add_argument("--cell", default=D115)
    c = sub.add_parser("curve")
    c.add_argument("run")
    c.add_argument("ref")
    c.add_argument("--band", default=None, help="STEP:DIFF,STEP:DIFF (default 500:0.05,2000:0.03,12000:0.02), e.g. 100:0.3,300:0.1,600:0.05 for the early twin check of a short run")
    sub.add_parser("selftest")
    return ap


def parse(argv):
    """argv -> namespace; the words after a lone `--` are the trainer's own arguments (`extra`); `--curve` needs `--ref`"""
    argv = list(argv)
    extra = []
    if "--" in argv:
        i = argv.index("--")
        argv, extra = argv[:i], argv[i + 1:]
    a = build_parser().parse_args(argv)
    a.extra = extra
    if a.cmd == "run" and a.curve is not None and not a.ref:
        raise SystemExit("--curve compares the arm with a reference run: give --ref")
    return a


def main():
    a = parse(sys.argv[1:])
    if a.cmd == "selftest":
        return selftest()
    if a.cmd == "ids":
        listed = card_run_ids(a.id)
        for arm in a.arm:
            for x in Arm(a.id, arm, 0, 0, a.ref, dry=True).step_ids():
                print(x + ("" if x in listed else "    (not in the card)"))
        return 0
    if a.cmd == "table":
        return table(a.runs, a.cell)
    if a.cmd == "curve":
        ok, text = curve(a.run, a.ref, parse_band(a.band))
        print(text)
        sys.exit(0 if ok else 1)
    arm = Arm(a.id, a.arm, a.seed, a.head_seed, a.ref, a.trainer, " ".join(a.extra), a.window, a.dry, curve_check=a.curve)
    arm.check_ids(card_run_ids(a.id))
    if not a.dry and not arm.lock():
        arm.say("another jarm runs this arm (lock /tmp/jarm_" + arm.run + ".lock): nothing to do")
        sys.exit(0)
    sys.exit(arm.go())


if __name__ == "__main__":
    main()
