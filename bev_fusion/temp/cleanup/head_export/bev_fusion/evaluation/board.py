#!/usr/bin/env python3
"""Board: every arm against the base chain, all heads at once, read from the stored evaluation files (runs/<name>/eval/metrics.json).

One row per arm, one column per head (maps, objects, motion, time shift). The base row holds the absolute values, an arm row the change
against the base mean. A name stands for all its seeds (runs/<name>_s0, _s1, ...) or for the run of exactly that name. No number is typed
here: every cell is a key of metrics.json.

    python evaluation/board.py                                  base and open arms from records/champion.json
    python evaluation/board.py t8_ctl t8_bevfeat t7_res4        the first name is the base
    python evaluation/board.py --write                          also write records/BOARD.md
    python evaluation/board.py --runs                           the evaluated runs with GT signature, split, memory mode and code commit
    python evaluation/board.py --selftest                       checks against the stored files of real runs (needs t8_ctl_s0, t8_bevfeat_s0)

Markers: ^ better and v worse than the band in every seed, ~ inside the band, ? the mean is beyond the band but the seeds disagree.
Band per column = max(0.02, 2 x spread of the base seeds). An arm is compared only when GT signature, split and memory mode equal the
base's (--force shows it anyway); a different code commit, number of frames or checkpoint is flagged (the plain metrics.json is the latest eval of any
--n, so 200 frames can stand next to 1000). The board is evidence: the decision is the card's rule (decide.py).
"""
import argparse
import datetime
import glob
import json
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RUNS = os.path.join(ROOT, "runs")
CHAMPION = os.path.join(ROOT, "records", "champion.json")
BOARD = os.path.join(ROOT, "records", "BOARD.md")
FLOOR = 0.02

SEG = "seg_fov/seg/dt+0.00/"
COLUMNS = [  # (group, label, metrics.json key, better)
    ("maps", "drivable", SEG + "drivable_area/all/all/iou@0.5", "high"),
    ("maps", "walkway", SEG + "walkway/all/all/iou@0", "high"),
    ("maps", "carpark", SEG + "carpark_area/all/all/iou@0", "high"),
    ("maps", "crossing", SEG + "ped_crossing/all/all/iou@0", "high"),
    ("maps", "lane div.", SEG + "lane_divider/all/all/iou@0.5", "high"),
    ("maps", "road div.", SEG + "road_divider/all/all/iou@1", "high"),
    ("objects", "mAP", "det/dt+0.00/mean/all/mAP", "high"),
    ("objects", "veh AP@2", "det/dt+0.00/vehicle/all/ap@2", "high"),
    ("objects", "ped AP@2", "det/dt+0.00/pedestrian/all/ap@2", "high"),
    ("motion", "veh AVE", "det/dt+0.00/vehicle/all/ave", "low"),
    ("time", "drivable +0.5 s", "seg_fov/seg/dt+0.50/drivable_area/all/all/iou@0.5", "high"),
]


def read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def run_dirs(name):
    """the run of exactly this name, else its seed runs <name>_s<N> in seed order"""
    exact = os.path.join(RUNS, name)
    if os.path.isdir(exact):
        return [exact]
    found = [p for p in glob.glob(exact + "_s*") if os.path.isdir(p) and re.fullmatch(r"s\d+", os.path.basename(p)[len(name) + 1:])]
    return sorted(found, key=lambda p: int(os.path.basename(p).rsplit("_s", 1)[1]))


class Run:
    def __init__(self, path):
        self.path = path
        self.name = os.path.basename(path)
        self.m = read_json(os.path.join(path, "eval", "metrics.json"))
        self.metrics = (self.m or {}).get("metrics") or {}
        self.identity = (self.m.get("gt_signature"), self.m.get("split"), self.m.get("memory")) if self.m else None
        self.frames = self.metrics.get("info/n_samples")      # the plain eval/metrics.json is the latest eval of any --n: its frame count and checkpoint are part of what a number means
        self.checkpoint = (self.m or {}).get("checkpoint")
        self.commit = self._commit()

    def _commit(self):
        try:
            with open(os.path.join(self.path, "code", "SNAPSHOT.txt")) as f:
                hit = re.match(r"commit (\w+)", f.readline())
            return hit.group(1) if hit else None
        except OSError:
            return None

    def state(self):
        if self.m:
            return "evaluated"
        if os.path.exists(os.path.join(self.path, ".done")):
            return "trained, not evaluated"
        if os.path.exists(os.path.join(self.path, "train.log")):
            return "training or stopped"
        return "no run"

    def value(self, key):
        v = self.metrics.get(key)
        return float(v) if isinstance(v, (int, float)) and math.isfinite(v) else None


class Group:
    def __init__(self, name):
        self.name = name
        self.runs = [Run(p) for p in run_dirs(name)]
        self.launcher = bool(glob.glob(os.path.join(RUNS, name + "*.launcher.log")))
        self.records = []                        # run.py records whose command trains a config of this name: (id, status)
        for p in glob.glob(os.path.join(ROOT, "records", "runs", "*", "run.json")):
            rec = read_json(p) or {}
            if re.search(r"configs/%s(_s\d+)?\.json" % re.escape(name), rec.get("command", "")):
                self.records.append((rec.get("id"), rec.get("status")))

    def evaluated(self):
        return [r for r in self.runs if r.m]


def mean(xs):
    return sum(xs) / len(xs)


def band_of(base_values, floor=FLOOR):
    return floor if len(base_values) < 2 else max(floor, 2 * (max(base_values) - min(base_values)))


def mark(deltas, band, better):
    """^ better, v worse in every seed beyond the band (two or more seeds); ? mean beyond the band but the seeds disagree or there is only one
    seed; ~ inside the band"""
    good = [d if better == "high" else -d for d in deltas]
    if len(good) < 2:
        return "?" if abs(mean(good)) > band else "~"
    if all(g > band for g in good):
        return "^"
    if all(g < -band for g in good):
        return "v"
    return "?" if abs(mean(good)) > band else "~"


def same_code(a, b):
    return not (a and b) or a.startswith(b) or b.startswith(a)


def setup_flags(base_runs, arm_runs, who=("arm", "base")):
    """', FRAMES DIFFER (arm 200, base 1000)' / ', CHECKPOINT DIFFERS (...)': what the two sets of evals were measured on, where both record it (flagged like a
    different code commit, not refused: the cells are shown); '' when equal or not recorded. who: the words for the two sets (the base row compares its first
    seed with its other seeds)"""
    out = ""
    for label, attr in (("FRAMES DIFFER", "frames"), ("CHECKPOINT DIFFERS", "checkpoint")):
        b, a = ({getattr(r, attr) for r in rs} - {None} for rs in (base_runs, arm_runs))
        if a and b and a != b:
            out += ", %s (%s %s, %s %s)" % (label, who[0], "/".join(sorted(map(str, a))), who[1], "/".join(sorted(map(str, b))))
    return out


def compat(base, arm):
    """(comparable, unrecorded fields): a field recorded on both sides must be equal; an unrecorded one (older evals) is a warning, as in bev_suite compare"""
    unknown = []
    for label, x, y in zip(("GT signature", "split", "memory mode"), base, arm):
        if x is not None and y is not None and x != y:
            return False, []
        if x is None or y is None:
            unknown.append(label)
    return True, unknown


def build(names, floor=FLOOR, force=False):
    """-> dict(base=..., bands=[...], rows=[dict(name, state, cells, deltas, marks, reading)])"""
    base = Group(names[0])
    base_ev = base.evaluated()
    if not base_ev:
        raise SystemExit("base %s: no evaluated run (state: %s)" % (names[0], ", ".join(r.state() for r in base.runs) or "no run folder"))
    ident = base_ev[0].identity
    if any(r.identity != ident for r in base_ev):
        raise SystemExit("base %s: its seeds differ in GT signature / split / memory: %s" % (names[0], [r.identity for r in base_ev]))
    base_vals = [[r.value(k) for r in base_ev if r.value(k) is not None] for _, _, k, _ in COLUMNS]
    bands = [band_of(v, floor) for v in base_vals]
    base_mean = [mean(v) if v else None for v in base_vals]
    base_commit = base_ev[0].commit
    rows = [dict(name=names[0] + " (base)", state="%d seed(s), %s%s" % (len(base_ev), base_commit or "commit ?", setup_flags(base_ev[:1], base_ev[1:], ("other seeds", "first seed"))),
                 cells=["%.3f" % m if m is not None else "-" for m in base_mean], deltas=[None] * len(COLUMNS),
                 marks=[""] * len(COLUMNS), reading="")]
    for name in names[1:]:
        g = Group(name)
        ev = g.evaluated()
        usable = [r for r in ev if compat(ident, r.identity)[0] or force]
        if not usable:
            if ev:
                why = "GT/split/memory differs from the base: %s" % (ev[0].identity,)
            elif g.runs:
                why = "; ".join("%s: %s" % (r.name, r.state()) for r in g.runs)
            else:
                why = "no run folder" + (", launcher log only" if g.launcher else "") + "".join("; run.py record %s: %s" % rec for rec in g.records)
            rows.append(dict(name=name, state=why, cells=["-"] * len(COLUMNS), deltas=[None] * len(COLUMNS), marks=[""] * len(COLUMNS), reading=""))
            continue
        cells, deltas, marks = [], [], []
        for i, (_, _, key, better) in enumerate(COLUMNS):
            vals = [r.value(key) for r in usable if r.value(key) is not None]
            if not vals or base_mean[i] is None:
                cells.append("-"); deltas.append(None); marks.append("")
                continue
            per_seed = [v - base_mean[i] for v in vals]
            mk = mark(per_seed, bands[i], better)
            cells.append("%+.3f %s" % (mean(per_seed), mk)); deltas.append(mean(per_seed)); marks.append(mk)
        up = [COLUMNS[i][1] for i, m in enumerate(marks) if m == "^"]
        down = [COLUMNS[i][1] for i, m in enumerate(marks) if m == "v"]
        unclear = [COLUMNS[i][1] for i, m in enumerate(marks) if m == "?"]
        reading = "%d better, %d worse" % (len(up), len(down)) + (" (%s)" % ", ".join(down) if down else "") + (", %d unclear" % len(unclear) if unclear else "")
        code = "" if all(same_code(r.commit, base_commit) for r in usable) else ", CODE DIFFERS (arm %s, base %s)" % (", ".join(sorted({r.commit or "?" for r in usable})), base_commit)
        state = "%d of %d seed(s)%s%s%s" % (len(usable), len(g.runs), ", %s" % usable[0].commit if usable[0].commit else "", code, setup_flags(base_ev, usable))
        if force and any(not compat(ident, r.identity)[0] for r in usable):
            state += ", FORCED: GT/split/memory differs"
        unrecorded = sorted({u for r in usable for u in compat(ident, r.identity)[1]})
        if unrecorded:
            state += ", not recorded in the eval: " + ", ".join(unrecorded)
        rows.append(dict(name=name, state=state, cells=cells, deltas=deltas, marks=marks, reading=reading))
    return dict(base=names[0], ident=ident, bands=bands, rows=rows, base_runs=[r.name for r in base_ev])


def champion():
    return read_json(CHAMPION) or {}


def render(board, champ):
    prom = champ.get("promoted")
    head = ["# Board - arms against the base, all heads (evaluation/board.py, %s)" % datetime.date.today().isoformat(), "",
            "Base `%s` (%s) | GT signature %s | split %s | memory %s | champion file: %s" % (
                board["base"], ", ".join(board["base_runs"]), board["ident"][0], board["ident"][1], board["ident"][2],
                "promoted %s" % prom.get("date") if isinstance(prom, dict) else "proposal, not promoted"),
            "", "Base row = mean value; arm rows = change against the base mean. Markers: ^ better and v worse than the band in every seed, "
            "~ inside the band, ? mean beyond the band but the seeds disagree or the arm has one seed. Lower is better: " +
            ", ".join(c[1] for c in COLUMNS if c[3] == "low") + ".",
            "Band per column (max of %.2f and 2 x base seed spread): " % FLOOR + ", ".join("%s %.3f" % (c[1], b) for c, b in zip(COLUMNS, board["bands"])) + ".",
            ("Base has ONE seed: the band is the floor, not a measured spread. " if len(board["base_runs"]) < 2 else "") +
            "This table is evidence. The decision is the card's rule (`decide.py`).", ""]
    lines = ["| arm | state | " + " | ".join("%s: %s" % (c[0], c[1]) for c in COLUMNS) + " | reading |",
             "|" + "---|" * (len(COLUMNS) + 3)]
    for r in board["rows"]:
        lines.append("| %s | %s | %s | %s |" % (r["name"], r["state"], " | ".join(r["cells"]), r["reading"]))
    return "\n".join(head + lines) + "\n"


def list_runs():
    print("%-26s %-22s %-18s %-6s %-14s %-7s %-18s %s" % ("run", "state", "gt_signature", "split", "memory", "frames", "checkpoint", "code"))
    for p in sorted(glob.glob(os.path.join(RUNS, "*"))):
        if os.path.isdir(p) and os.path.exists(os.path.join(p, "eval", "metrics.json")):
            r = Run(p)
            print("%-26s %-22s %-18s %-6s %-14s %-7s %-18s %s" % (r.name, r.state(), r.identity[0], r.identity[1], r.identity[2], r.frames, r.checkpoint, r.commit))


def selftest_setup(check):
    """frames / checkpoint flags on synthetic stored evals (a throw-away runs folder): an arm evaluated on 200 frames or at another checkpoint stands next to the base
    with its cells and a flag in the state; equal setups carry no flag"""
    global RUNS
    import tempfile
    keep = RUNS
    with tempfile.TemporaryDirectory() as d:
        RUNS = d
        try:
            for name, frames, ckpt in (("base", 1000, "ckpt_00003000.pt"), ("same", 1000, "ckpt_00003000.pt"), ("short", 200, "ckpt_00003000.pt"),
                                       ("early", 1000, "ckpt_00001000.pt")):
                os.makedirs(os.path.join(d, name + "_s0", "eval"))
                with open(os.path.join(d, name + "_s0", "eval", "metrics.json"), "w") as f:
                    json.dump({"gt_signature": "g", "split": "val", "memory": "empty (cold)", "checkpoint": ckpt,
                               "metrics": {COLUMNS[0][2]: 0.5, "info/n_samples": frames}}, f)
            rows = {r["name"]: r for r in build(["base", "same", "short", "early"])["rows"]}
        finally:
            RUNS = keep
    check("a different number of frames / checkpoint is flagged with both values, the cells stay; equal setups carry no flag",
          "FRAMES DIFFER (arm 200, base 1000)" in rows["short"]["state"] and "CHECKPOINT DIFFERS" in rows["early"]["state"] and rows["short"]["cells"][0] != "-"
          and "FRAMES" not in rows["early"]["state"] and "DIFFER" not in rows["same"]["state"] and "DIFFER" not in rows["base (base)"]["state"])


def selftest():
    ok = []

    def check(name, cond):
        ok.append(bool(cond))
        print("%s  %s" % ("PASS" if cond else "FAIL", name))
    selftest_setup(check)
    need = [os.path.join(RUNS, n, "eval", "metrics.json") for n in ("t8_ctl_s0", "t8_bevfeat_s0")]
    missing = [p for p in need if not os.path.exists(p)]
    if missing:
        print("selftest needs the stored runs:", missing)
        return 1

    ctl, bev = Run(os.path.join(RUNS, "t8_ctl_s0")), Run(os.path.join(RUNS, "t8_bevfeat_s0"))
    table = {}
    with open(os.path.join(RUNS, "t8_ctl_s0", "eval", "table.txt")) as f:
        for line in f:
            parts = line.split()
            if len(parts) == 2:
                table[parts[0]] = float(parts[1])
    check("every board key of t8_ctl_s0 equals the independent eval/table.txt (4 decimals)",
          all(abs(ctl.value(k) - table[k]) < 6e-5 for _, _, k, _ in COLUMNS if ctl.value(k) is not None and k in table) and sum(k in table for _, _, k, _ in COLUMNS) >= 9)
    card_ctl = [0.716, 0.178, 0.115, 0.002, 0.191, 0.284]
    card_bev = [0.804, 0.339, 0.173, 0.145, 0.351, 0.501]
    check("t8_ctl_s0 maps equal the numbers in experiments/t8_bevfeat/experiment.md", all(abs(ctl.value(COLUMNS[i][2]) - card_ctl[i]) < 6e-4 for i in range(6)))
    check("t8_bevfeat_s0 maps equal the numbers in experiments/t8_bevfeat/experiment.md", all(abs(bev.value(COLUMNS[i][2]) - card_bev[i]) < 6e-4 for i in range(6)))
    fwd, rev = build(["t8_ctl_s0", "t8_bevfeat_s0"]), build(["t8_bevfeat_s0", "t8_ctl_s0"])
    pairs = [(a, b) for a, b in zip(fwd["rows"][1]["deltas"], rev["rows"][1]["deltas"]) if a is not None and b is not None]
    check("swapping base and arm negates all %d changes" % len(COLUMNS), len(pairs) == len(COLUMNS) and all(abs(a + b) < 1e-9 for a, b in pairs))
    row = fwd["rows"][1]
    check("t8_bevfeat_s0 (one seed): six maps beyond the band upwards, mAP downwards, all marked ? because one seed proves nothing",
          row["marks"][:7] == ["?"] * 7 and all(d > 0 for d in row["deltas"][:6]) and row["deltas"][6] < 0)
    check("marker rules: both seeds up, seeds disagree, inside the band, lower-is-better flip",
          mark([0.05, 0.04], 0.02, "high") == "^" and mark([0.10, 0.01], 0.02, "high") == "?" and mark([0.01, -0.01], 0.02, "high") == "~"
          and mark([0.05], 0.02, "low") == "?" and mark([-0.05], 0.02, "low") == "?" and mark([0.05, 0.04], 0.02, "low") == "v")
    check("a name without a run is pending, not a number", build(["t8_ctl_s0", "zz_no_such_run"])["rows"][1]["cells"] == ["-"] * len(COLUMNS))
    runs = [Run(p) for p in sorted(glob.glob(os.path.join(RUNS, "*"))) if os.path.exists(os.path.join(p, "eval", "metrics.json"))]
    other = next((r for r in runs if not compat(ctl.identity, r.identity)[0]), None)
    if other:
        row = build(["t8_ctl_s0", other.name])["rows"][1]
        check("an evaluated run with another GT signature (%s) is not compared" % other.name, row["cells"] == ["-"] * len(COLUMNS) and "differs" in row["state"])
        forced = build(["t8_ctl_s0", other.name], force=True)["rows"][1]
        check("--force shows it and says so", "FORCED" in forced["state"] and forced["cells"][0] != "-")
    else:
        print("SKIP  no evaluated run with another GT signature found")
    print("%d of %d checks passed" % (sum(ok), len(ok)))
    return 0 if all(ok) else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="*", help="base first, then the arms (default: records/champion.json)")
    ap.add_argument("--write", action="store_true", help="write records/BOARD.md")
    ap.add_argument("--band", type=float, default=FLOOR)
    ap.add_argument("--force", action="store_true", help="also show arms whose GT signature / split / memory mode differs")
    ap.add_argument("--runs", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    if a.runs:
        return list_runs()
    champ = champion()
    names = a.names or ([champ["base"]] + list(champ.get("open", [])) if champ.get("base") else [])
    if not names:
        sys.exit("no names given and no base in records/champion.json")
    text = render(build(names, a.band, a.force), champ)
    print(text)
    if a.write:
        with open(BOARD, "w") as f:
            f.write(text)
        print("written", BOARD)


if __name__ == "__main__":
    main()
