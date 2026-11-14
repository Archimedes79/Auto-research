"""The one-shot evaluation of a finished lane system on a SPLIT of the project (VAL for the checks, TEST for the logged uses: E321 = E312, D84; E343 = E324 s with the stack E338, D95): B's prediction dump, the pair head's dump, the delivered graphs of the
read-outs of the system (a `SETUPS` entry), the topology numbers of the cards, the maneuver coverage of E310, the paired scene bootstrap and, for E324 s, the comparison of every arm with E312 and the reading of the E343 card.

    wsl bash tools/env/_py14.sh jtest_eval.py dump-b    --split val --seed 0 [--b-run runs/e292_c2_s0] [--device cpu] [--out PATH] [--max N]
    wsl bash tools/env/_py14.sh jtest_eval.py dump-ends --split val --seed 0 [--run e324_s_s0] [--device cpu] [--out PATH] [--max N]
    wsl bash tools/env/_py14.sh jtest_eval.py all       --split val --seeds 0 1 [--run-prefix e324_s_s] [--device cpu] [--b-dump P] [--control-ends-dump P] [--control-card P] [--no-stack]   # what run.py launches for a TEST use
    wsl bash tools/env/_py14.sh jtest_eval.py verify    --seed 0 | --run e324_s_s0 [--no-stack]       # VAL, from the stored dumps: reproduce the stored cards exactly (E312: 3 cards + maneuver_val.json; E324 s: its 3 read-outs + the stack card)
    wsl bash tools/env/_py14.sh jtest_eval.py selftest                                                  # and tests/test_jtest_eval.py

Systems (`SETUPS`, chosen by `--run-prefix` / the run name): `e312_p_s` (plain, connectors not snapped, snapped) and `e324_s_s` (plain, snapped = the card of record of D94, snap_nogeom, primary = attached ends + snap_nogeom, and the stack E338: the same pair head on
the delivered graph of c2 = `runs/e292_c2_s<S>`, snapped). The functions of `jends_eval.py`, `qends_eval.py`, `jmaneuver.py`, `qgraph_eval.py` are used as they are; with `--split val`, `verify` reproduces their stored cards (the split is the only difference).

`all` per seed: the dumps (a dump that exists is VALIDATED and reused, never re-dumped or overwritten; `--b-dump` names B's dump to reuse and it must exist), the scoring of every variant in one pass, then the cards and `results.md` of both seeds.
Before any dump is made, `all` checks the seeds of the pair runs, the pixel pyramid of the stack, the dumps named by flags and the control card (refused there, not after hours). A dump that this module made (default paths) has a sidecar `<name>_head.json` with
its provenance (run, device, sha256 of the checkpoint, digest of the imported lane code): an existing one is reused only under the same provenance, else refused with a message (nothing is deleted: move it away). Dumps named by a flag (E321's, the stored VAL
dumps) are exempt: validated, and E321's also by hash. Cards and `results.md` are written once (refused when they exist; `--max` runs excepted); they name the device and carry the sha256 of the imported lane code, and `CODE DRIFT: <files>` is printed and
written to `results.md` when one of those files changed during the run (not fatal). A standalone `dump-b` / `dump-ends` never overwrites a dump, and a name without `.pkl` is refused (the head file would overwrite it).
The control (E324 s only) is E312's stored card (`--control-card`, default per split: TEST `runs/eval/e312_test/card_s{seed}.json`, VAL `runs/e312_p_s{seed}/ends_card_snap_thr0.1.json`): its numbers and its plain graph are the reference; the plain graph computed
here must equal the card's (TEST 1e-9, VAL 3e-3 for a fresh CPU dump) or `all` stops before any number is printed. `--control-ends-dump` (default only on VAL: the stored E312 dump) re-scores E312's pair head for its per-frame counts, which give the paired
interval; it is used only when it reproduces the stored card to 1e-9 on EVERY seed, else there is no interval on any seed and `reading()` takes its fallback. `{seed}` in a path flag is the seed.

THE DRESS REHEARSAL ON VAL (criterion 2 of the card E343): the TEST command with VAL files, on the CPU, for S = 0 and 1. `verify --seed S` writes `temp/e343/val/val_check/verify_card_e312_p_s<S>.json`, E312's snapped card from the stored dumps in the layout of
E321's `card_s<S>.json`. The plain-graph tolerance of the rehearsal is the loose VAL one by design (a fresh CPU dump against GPU cards); the 1e-9 branch of the TEST kill criterion is covered by a toy test.
  (a) control from the stored VAL files (the TEST layout of the control card, the flagged B dump; E312's counts do NOT reproduce on the fresh B graph, so there is no interval and `reading()` takes its fallback: expected):
      jtest_eval.py dump-b --split val --seed S                                          # B's fresh dump -> temp/e343/val/predfull_e273_p_s<S>.pkl
      jtest_eval.py all --split val --test-id E343.v --run-prefix e324_s_s --seeds 0 1 --device cpu --b-dump temp/e343/val/predfull_e273_p_s{seed}.pkl --control-ends-dump runs/e312_p_s{seed}/ends_val.pkl --control-card temp/e343/val/val_check/verify_card_e312_p_s{seed}.json
  (b) control from the SAME fresh dumps (as on TEST, where E321's dumps and card belong together: E312's counts reproduce to 1e-9, the intervals are used):
      jtest_eval.py dump-b --split val --seed S;  jtest_eval.py dump-ends --split val --seed S --run e312_p_s<S>                # temp/e343/val/predfull_e273_p_s<S>.pkl, ends_e312_p_s<S>.pkl
      jtest_eval.py all --split val --run-prefix e312_p_s --seeds 0 1 --b-dump temp/e343/val/predfull_e273_p_s{seed}.pkl         # E312's cards of these dumps -> runs/eval/e312_val/card_s<S>.json (E321's layout), checked against its stored cards
      jtest_eval.py all --split val --test-id E343.v --run-prefix e324_s_s --seeds 0 1 --device cpu --b-dump temp/e343/val/predfull_e273_p_s{seed}.pkl --control-ends-dump temp/e343/val/ends_e312_p_s{seed}.pkl --control-card runs/eval/e312_val/card_s{seed}.json

THE TEST GATE (hard rule 1, D84). `--split test` is refused unless (1) `--test-id <run id>` is given, (2) the process runs under the launcher (`AR_RUN_ID` = that id, set by `autoresearch/tools/run.py`), (3) the TEST log (`evaluation/TEST_USAGE_LOG.md`) opens the split for the id by
the gate that `evaluate.py` and `run.py` share (`autoresearch/tools/testgate.py`), and (4) `--max` is not used. Nothing here selects, tunes or re-reads anything on the split: the read-outs, the thresholds (p >= 0.1), the close-node step `close:1.5@0.5` and the metrics are the ones of the VAL cards.

Output (ONE write helper, `atomic`: a temporary file renamed at the end, refusing every path under `temp/e321/` and `runs/eval/e312_test/`, E321's artefacts, which are read-only inputs and are hashed at the start and the end of `all`): dumps in `temp/e343/<split>/`
(`predfull_<B run>.pkl`, `ends_<run>.pkl`), cards in `runs/eval/e324s_<split>/` (E312: `e312_<split>`) as `card_s<S>.json` + `results.md`; a smoke run (`--max N`, never on TEST) writes under `temp/e343/<split>_max<N>/` only. Every dump on the path is validated: the split's frame count and its rows in order.
The graph GT of the frames is `graph_gt_val.pkl[row]` (that file, the tokens `val_tokens.json`, `poses_val.npz` and the features hold the VAL and the TEST rows of the cache; the rows of a split are `data/splits.json[split]`), the lane GT is the l6 store.
"""
import argparse
import ast
import contextlib
import hashlib
import json
import os
import pickle
import re
import sys
import time

import numpy as np

ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for sub in ("evaluation", "training", "training/grid", "training/data", "training/data/prep", "training/train", "tools/viz", ".."):
    sys.path.insert(0, os.path.normpath(os.path.join(ROOT, sub)))
sys.path.append(os.path.normpath(os.path.join(ROOT, "..", "autoresearch")))

SPLITS = {"val": "val_clean", "test": "test_clean"}
EXPECT_FRAMES = {"val": 873, "test": 833}
THR = 0.1                                                                                         # the delivery threshold of E309 / E312 (fixed on VAL)
CLOSE = "close:1.5@0.5"                                                                           # the named close-node step of E275 (fixed on VAL)
KEYS = ("Q", "F1@1.0", "lanes_found", "topo_F1", "topo_P", "topo_R", "junction_F1", "junction_P", "junction_R", "dup_frac", "vert100_dp")
LOWER = ("dup_frac", "vert100_dp")
EXACT = dict(topo=1e-9, real=1e-9, count_rel=0.0)                                                 # criterion 1 of E343 (stored dumps) and the TEST path
REHEARSAL = dict(topo=1e-3, real=3e-3, count_rel=5e-4)                                            # criterion 2 of E343: a fresh CPU dump against the stored (GPU) VAL cards


def _require(cond, msg):
    """a check that stays under `python -O` (an `assert` does not)"""
    if not cond:
        raise RuntimeError(msg)


# ------------------------------------------------------------------------------------------------------------------------------ the read-outs of a system, as data
def V(name, ends=None, graph="b", snap=False, nogeom=False, attach=False, man=None, base=None):
    """one delivered graph. `graph`: the dump it is read out from (b = B's, c2 = the stack's, ctl = E312's graph); `ends` None = that plain graph (the close-node step only), else the connectors at p >= THR of the pair dump `ends` (own = the run's, ctl = the
    control's); snap / nogeom (`snap_nogeom`) / attach: the options of `qends_eval.add_connectors` / `make_pairs` (the stored cards' `--snap --snap-nogeom --attach`); `man`: its label in the maneuver table (None = no maneuver count);
    `base`: the plain variant it is compared with (D79 rule, paired bootstrap)"""
    return dict(name=name, ends=ends, graph=graph, snap=snap, nogeom=nogeom, attach=attach, man=man, base=base)


SETUPS = {
    "e312_p_s": dict(
        label="E312", out="e312", graph_run="e273_p_s{seed}",
        variants=[V("plain", man="plain"), V("unsnapped", "own", man="p0.1", base="plain"), V("snapped", "own", snap=True, man="p0.1s", base="plain")],
        stored={"unsnapped": "ends_card_thr0.1.json", "snapped": "ends_card_snap_thr0.1.json"}, stored_counts=("snapped",), stored_man="maneuver_val.json", stack=[], arms={}),
    "e324_s_s": dict(
        label="E324 s", out="e324s", graph_run="e273_p_s{seed}",
        variants=[V("plain", man="plain"), V("snapped", "own", snap=True, man="p0.1s", base="plain"), V("snap_nogeom", "own", snap=True, nogeom=True, base="plain"),
                  V("primary", "own", snap=True, nogeom=True, attach=True, man="p0.1an", base="plain")],
        stack=[V("c2_plain", graph="c2", man="c2 plain"), V("stack", "own", graph="c2", snap=True, man="c2 p0.1s", base="c2_plain")], stack_run="e292_c2_s{seed}",
        stored={"snapped": "ends_card_snap_thr0.1.json", "snap_nogeom": "ends_card_snap_nogeom_thr0.1.json", "primary": "ends_card_snap_nogeom_attach_thr0.1.json", "stack": "ends_card_e338_snap_thr0.1.json"},
        stored_counts=("snapped", "snap_nogeom", "primary", "stack"), stored_man=None, arms=dict(w="snapped", p="primary", k="stack")),
}
CONTROL_VARIANT = V("e312_snapped", "ctl", graph="ctl", snap=True)                                  # E312's snapped read-out on E312's own dumps (its per-frame counts, from which the paired intervals come)
STORED_DUMPS = dict(b="temp/e276/predfull_e273_p_s{seed}.pkl", c2="temp/e292/predfull_e292_c2_s{seed}.pkl")  # the VAL dumps that made the stored cards
CONTROL_DEFAULTS = {"val": dict(ends="runs/e312_p_s{seed}/ends_val.pkl", b=STORED_DUMPS["b"], card="runs/e312_p_s{seed}/ends_card_snap_thr0.1.json"),
                    "test": dict(ends=None, b=None, card="runs/eval/e312_test/card_s{seed}.json")}   # TEST: the control's graph is the same B dump; the control's pair dump only by flag (the reading R1 of the card E343)


def setup_of(run):
    """(setup, seed) of a run name such as e324_s_s0"""
    for prefix, st in SETUPS.items():
        m = re.fullmatch(re.escape(prefix) + r"(\d+)", run)
        if m:
            return st, int(m.group(1))
    raise SystemExit(f"no setup for the run {run!r} (prefixes: {', '.join(SETUPS)})")


def variants_of(setup, stack=True, control=False):
    return list(setup["variants"]) + (list(setup["stack"]) if stack else []) + ([CONTROL_VARIANT] if control else [])


# ------------------------------------------------------------------------------------------------------------------------------ the gate
def test_gate_open(test_id, env_run_id, log=None, decided=None):
    """(ok, why): the TEST split opens only for a logged id under the launcher; `log` / `decided` are for the tests (default: the project's log and its granting decisions, as evaluate.py)"""
    if not test_id:
        return False, "TEST refused: pass --test-id <run id> (the id of the run.py launch that the TEST log names)"
    if env_run_id != test_id:
        return False, f"TEST refused: the TEST split runs only under the launcher (`run.py --id {test_id} --split test -- ...`: AR_RUN_ID is '{env_run_id}', not '{test_id}')"
    import evaluate as EV                                                                            # the lane's evaluator: the same gate function, loaded only for TEST
    if not EV.test_allowed(test_id, log=log, decided=decided):
        return False, f"TEST refused: {test_id} is not opened by evaluation/TEST_USAGE_LOG.md (a table row naming the id with the person's OK)"
    return True, ""


def require_gate(a):
    if a.split == "test":
        if getattr(a, "max", 0):
            sys.exit("TEST refused: --max (a truncated run) is for smoke tests on VAL; the TEST split is read whole")
        ok, why = test_gate_open(a.test_id, os.environ.get("AR_RUN_ID", ""))
        if not ok:
            sys.exit(why)


# ------------------------------------------------------------------------------------------------------------------------------ output: the ONE write helper, E321's artefacts
PROTECTED = ("temp/e321", "runs/eval/e312_test")                                                    # E321's dumps and cards: read-only inputs of E343
E321_SHA256 = {"temp/e321/test/predfull_e273_p_s0.pkl": "5f0500e8dd251f7966b0629223b5c428c8991f35d2637dbf0ce7a06e846970df", "temp/e321/test/predfull_e273_p_s1.pkl": "fa87972eb1aad01adc18dede7c8632335e6b9c2be73fb68b6d26605fa3510d28",
               "temp/e321/test/ends_e312_p_s0.pkl": "8b1746d8927a51dc3fec2db8a27f47d114dde966a9a1588af3de816cff58c118", "temp/e321/test/ends_e312_p_s1.pkl": "5ff2142df247692d6af46b670c0a9d9e4fbe4c9883879a37388e7ada59cd5700",
               "runs/eval/e312_test/card_s0.json": "1f9628c00ca3a86e2e3e6791b259f86599d6291839f596a9f2fa7b6f5c82c049", "runs/eval/e312_test/card_s1.json": "6aa987ab74f1eb1207e524da39fd66752221d42d5cf3e8091cbbee6bcab7ec3b",
               "runs/eval/e312_test/results.md": "113871898d86f9d639b6b6586947ca61c510b31f882c1961f603bf0f594e4b13"}


def is_protected(path):
    """True when the real path (symlinks and `..` resolved, upper / lower case ignored: /mnt/c is case-insensitive) lies under one of the PROTECTED folders"""
    real = os.path.realpath(path).casefold()
    return any(real == r or real.startswith(r + os.sep) for r in (os.path.realpath(os.path.join(ROOT, p)).casefold() for p in PROTECTED))


@contextlib.contextmanager
def atomic(path, overwrite=True):
    """the ONE way this module writes a dump or a card: yields a temporary file name next to `path`, renames it onto `path` when the block ends (a crash leaves no truncated file that a re-run could reuse); refuses a protected path before anything is created,
    and with `overwrite=False` a file that exists (a result is written once)"""
    if is_protected(path):
        raise PermissionError(f"refused: {path} lies under {' / '.join(PROTECTED)} (the artefacts of E321 are read-only inputs)")
    if not overwrite and os.path.exists(path):
        raise FileExistsError(f"refused: {path} exists (a result is written once: move it away to write it again)")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}"
    try:
        yield tmp
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def save_pickle(path, obj, overwrite=True):
    with atomic(path, overwrite) as tmp:
        with open(tmp, "wb") as f:
            pickle.dump(obj, f)


def save_json(path, obj, overwrite=True):
    save_text(path, json.dumps(obj, indent=1), overwrite)


def save_text(path, text, overwrite=True):
    with atomic(path, overwrite) as tmp:
        with open(tmp, "w") as f:
            f.write(text)


def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


CODE_SKIP = ("evaluate", "GtGenerator")                                                             # the TEST gate's evaluator pulls in the whole evaluation suite; the GT store is keyed by its own signature


def _imports(path):
    """(level, module name) of every import statement of a source file; `from m import n` also as m.n (n may be a submodule)"""
    try:
        with open(path, encoding="utf-8") as f:
            tree = ast.parse(f.read())
    except (OSError, SyntaxError, UnicodeDecodeError):
        return []
    out = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            out += [(0, a.name) for a in n.names]
        elif isinstance(n, ast.ImportFrom):
            base = n.module or ""
            out += [(n.level, base)] + [(n.level, f"{base}.{a.name}" if base else a.name) for a in n.names]
    return out


def code_sha256(entry=None):
    """sha256 of every file of the lane code this module runs, keyed by the path from the repository root: the static closure of the `import` statements of `entry` (default: this file) over the project's code folders, found the way `sys.path` finds them
    (the lane's evaluation / training folders, the repository root, autoresearch/, so `Evaluation/stats.py` is in); modules of the library are not followed, nor those of CODE_SKIP"""
    dirs = [os.path.normpath(os.path.join(ROOT, p)) for p in ("evaluation", "training", "training/grid", "training/data", "training/data/prep", "training/train", "tools/viz", "..", "../autoresearch")]
    repo = os.path.realpath(os.path.join(ROOT, ".."))
    found, todo = set(), [os.path.realpath(entry or os.path.abspath(__file__))]
    while todo:
        p = todo.pop()
        if p in found:
            continue
        found.add(p)
        for level, name in _imports(p):
            if not name or name.split(".")[0] in CODE_SKIP:
                continue
            roots = [os.path.normpath(os.path.join(os.path.dirname(p), *([".."] * (level - 1))))] if level else dirs
            parts = name.split(".")
            for d in roots:
                base = os.path.join(d, *parts)
                hit = next((c for c in (base + ".py", os.path.join(base, "__init__.py")) if os.path.isfile(c)), None)
                if hit:
                    todo += [os.path.realpath(hit)] + [os.path.realpath(c) for c in (os.path.join(d, *parts[:k], "__init__.py") for k in range(1, len(parts))) if os.path.isfile(c)]          # the packages above it run too
                    break
    return {os.path.relpath(p, repo).replace(os.sep, "/"): sha256_file(p) for p in sorted(found)}


def code_digest(files):
    return hashlib.sha256("\n".join(f"{k} {v}" for k, v in sorted(files.items())).encode()).hexdigest()


def provenance(run, device):
    """what a dump of `run` made now depends on: the code digest, the device, the checkpoint (a dump is reused only under the same provenance)"""
    return dict(run=run, device=device, code_digest=code_digest(code_sha256()), checkpoint_sha256=sha256_file(os.path.join(ROOT, "runs", run, "model.pt")))


def head_path(out):
    _require(out.endswith(".pkl"), f"{out}: the name of a dump ends with .pkl (its head and provenance go to <name>_head.json, which must not overwrite it)")
    return out[: -len(".pkl")] + "_head.json"


def write_head(out, head, prov):
    save_json(head_path(out), dict(head, provenance=prov))


def check_provenance(path, expect, what):
    """a dump this module made earlier is reused only when its sidecar names the same run, code, device and checkpoint; else it is refused (nothing is deleted: move it away and `all` dumps again)"""
    hp = head_path(path)
    if not os.path.exists(hp):
        sys.exit(f"{what}: {path} has no provenance ({hp}): it was made by other code; move it away and run again")
    with open(hp) as f:
        got = json.load(f).get("provenance") or {}
    bad = [k for k in ("run", "code_digest", "device", "checkpoint_sha256") if got.get(k) != expect.get(k)]
    if bad:
        sys.exit(f"{what}: {path} was made under another {', '.join(bad)} (provenance {hp}): it is not reused; move it away and run again")


def refuse_existing(path):
    if os.path.exists(path):
        raise SystemExit(f"refused: {path} exists: a dump is never overwritten (`all` reuses a valid one; move this one away to dump again)")


def e321_changed():
    """the artefacts of E321 whose sha256 differs from the card's, and the ones that are missing (a deleted artefact is a change)"""
    out = []
    for rel, want in E321_SHA256.items():
        p = os.path.join(ROOT, rel)
        if not os.path.exists(p):
            out.append(f"{rel} (missing)")
        elif sha256_file(p) != want:
            out.append(rel)
    return out


def guard_e321(when):
    bad = e321_changed()
    if bad:
        sys.exit(f"STOP {when}: the sha256 of E321's artefact(s) differs from the card's: {bad}")


def dump_dir(split, n=0):
    return os.path.join(ROOT, "temp", "e343", split + (f"_max{n}" if n else ""))


def out_dir(setup, split, n=0):
    return os.path.join(dump_dir(split, n), "cards_" + setup["out"]) if n else os.path.join(ROOT, "runs", "eval", f"{setup['out']}_{split}")


# ------------------------------------------------------------------------------------------------------------------------------ frames and dumps
def split_rows(split):
    """the cache rows of a split (data/splits.json): ids only"""
    return np.asarray(json.load(open(f"{ROOT}/data/splits.json"))[split], int)


def frames_from_rows(rows):
    """Frame objects for cache rows: token (val_tokens.json), scene (poses_val.npz), the old graph GT (graph_gt_val.pkl); the files hold the VAL and the TEST rows of the cache, as `qgraph_eval.val_frames()` reads them"""
    import qgraph_eval as QE
    tokens = json.load(open(f"{QE.LS.B123}/feats/val_tokens.json"))
    scene = np.load(f"{QE.LS.B123}/poses_val.npz")["scene"]
    gt = pickle.load(open(f"{QE.LS.B123}/graph_gt_val.pkl", "rb"))
    _require(rows.max() < min(len(tokens), len(scene), len(gt)), (rows.max(), len(tokens), len(scene), len(gt)))
    return [QE.Frame(k, int(r), tokens[r], int(scene[r]), gt[r]) for k, r in enumerate(rows)]


def frames_of(split):
    """the frames of a split in the order of data/splits.json[split] (for val exactly `qgraph_eval.val_frames()`: the selftest checks it)"""
    rows = split_rows(split)
    _require(len(rows) == EXPECT_FRAMES[split], (split, len(rows)))
    return frames_from_rows(rows)


def validate_dump(items, split, what, n=0):
    """every dump on the path of a split holds the split's frames, all of them and in the order of data/splits.json (a smoke run, n > 0: the first n of them, of a longer dump too)"""
    rows = split_rows(split).tolist()
    _require(len(rows) == EXPECT_FRAMES[split], f"data/splits.json has {len(rows)} frames in {split}, expected {EXPECT_FRAMES[split]}")
    rows, items = rows[: n or None], items[: n or None]
    _require(len(items) == len(rows), f"{what}: {len(items)} frames, the split {split} has {len(rows)}")
    _require([int(d["row"]) for d in items] == rows, f"{what}: the rows are not the rows of the split {split} in the order of data/splits.json")
    return items


def load_dump(path, split, what, n=0):
    with open(path, "rb") as f:
        return validate_dump(pickle.load(f), split, f"{what} ({path})", n)


def get_dump(path, make, split, what, n=0, expect=None):
    """a dump of the split: read and validated when it exists (never overwritten or made again), else made by `make()` (None: the path was named by a flag, a missing file is an error). `expect`: the provenance of a dump this module made
    (the default paths): an existing one is refused unless its sidecar says the same run, code, device and checkpoint; a dump named by a flag (E321's, the stored VAL dumps: validated, E321's also by hash) has none and is exempt"""
    if not os.path.exists(path):
        if make is None:
            sys.exit(f"{what}: {path} does not exist (a dump named by a flag is reused, never made)")
        make()
    elif expect is not None:
        check_provenance(path, expect, what)
    return load_dump(path, split, what, n)


def dump_b(split, run, out, device="cpu", n=0):
    """the full prediction pass of a B-type run (runs/<run>: B = e273_p_s<S>, c2 = e292_c2_s<S>) on the split, exact map, as `qgraph_run_eval.score --dump-full`: the pickle `qgraph_readout.dump_full` writes"""
    head_path(out)                                                                                     # a dump name ends with .pkl; an existing dump is never overwritten
    refuse_existing(out)
    import lane_gt_data as LD
    import qgraph_readout as RO
    import qpredict as QP
    import torch
    frames = frames_of(split)
    m = QP.load_model(os.path.join(ROOT, "runs", run), device)
    ds, dl = LD.loader(SPLITS[split], bs=4, workers=0, shuffle=False, map_mode="exact", pixels=False, strict=True)
    rows = [int(ds.bd.rows[k]) for k in ds.ks]
    _require(rows == [f.row for f in frames], "the frame order of the loader differs from the scorer's")
    preds, t0 = [], time.time()
    with torch.no_grad():
        for b in dl:
            preds += QP.predict(m, b)
            if n and len(preds) >= n:
                break
            if len(preds) % 100 < 4:
                print(f"  dump-b {len(preds)}/{len(rows)}  {time.time() - t0:.0f} s", flush=True)
    preds = preds[: n or None]
    with atomic(out) as tmp:
        RO.dump_full(preds, rows[: len(preds)], tmp)
    write_head(out, dict(split=split, frames=len(preds)), provenance(run, device))
    print(f"dump-b {split} {run}: {len(preds)} frames -> {out}  {time.time() - t0:.0f} s", flush=True)


def dump_ends(split, run, out, device="cpu", n=0, thr=0.5):
    """the pair head's dump on the split, as `jends_eval.dump`: keypoints, pair logits, handle scales, and when the model has them the position scales (logsig) and the attached ends (att_pos / att_dir / att_p, E320); the junction paint sidecars are not needed:
    the model reads the features only, the GT ends come from the lane GT. The head file next to it holds the recall / precision of the pair head and the provenance (nothing derived from the split's labels is printed)"""
    head_path(out)
    refuse_existing(out)
    import jends_eval as JE
    import lane_gt_data as LD
    import qendnet as QEN
    import torch
    frames = frames_of(split)
    m, args = JE.load_model(run, device)
    ds, dl = LD.loader(SPLITS[split], bs=8, workers=0, shuffle=False, map_mode="error", pixels=False, ends=True, strict=True)
    rows = [int(ds.bd.rows[k]) for k in ds.ks]
    _require(rows == [f.row for f in frames], "the frame order of the loader differs from the scorer's")
    out_frames, tot, t0 = [], {}, time.time()
    with torch.no_grad():
        for b in dl:
            x = b["x"].to(device).float()
            _, _, oe = m(x, gt=None)
            for k, v in QEN.end_counts(oe, b, thr).items():
                tot[k] = tot.get(k, 0) + v
            for i in range(len(b["token"])):
                fr = dict(token=b["token"][i], row=int(b["row"][i]), kp_pos=oe["kp_pos"][i].cpu().numpy(), kp_dir=oe["kp_dir"][i].cpu().numpy(), kp_p=oe["kp_p"][i].cpu().numpy(),
                          valid=oe["valid"][i].cpu().numpy(), logit=oe["logit"][i].float().cpu().numpy(), allowed=oe["allowed"][i].cpu().numpy(),
                          scale=oe["scale"][i].float().cpu().numpy() if "scale" in oe else None)
                if "kp_logsig" in oe:                                                                # E318: the position scale (log sigma, m) of every keypoint
                    fr["logsig"] = oe["kp_logsig"][i].float().cpu().numpy()
                if "att_pos" in oe:                                                                  # E320: the attached ends (position, heading, probability of an attachment) of every keypoint
                    fr.update(att_pos=oe["att_pos"][i].float().cpu().numpy(), att_dir=oe["att_dir"][i].float().cpu().numpy(), att_p=oe["att_p"][i].float().cpu().numpy())
                out_frames.append(fr)
            if len(out_frames) % 80 == 0:
                print(f"  dump-ends {len(out_frames)}/{len(rows)}  {time.time() - t0:.0f} s", flush=True)
            if n and len(out_frames) >= n:
                break
    out_frames = out_frames[: n or None]
    r = lambda x, y: x / max(y, 1)  # noqa: E731
    head = dict(ent_recall=r(tot["ent_hit"], tot["ent_n"]), ent_precision=r(tot["ent_ok"], tot["ent_kp"]), ext_recall=r(tot["ext_hit"], tot["ext_n"]), ext_precision=r(tot["ext_ok"], tot["ext_kp"]),
                pair_recall=r(tot["pairs_hit"], tot["pairs_n"]), pair_precision=r(tot["pairs_ok"], tot["pairs_kp"]), counts=tot, thr=thr, frames=len(out_frames), run=run)
    save_pickle(out, out_frames)
    write_head(out, head, provenance(run, device))
    print(f"dump-ends {split} {run}: {len(out_frames)} frames -> {out}  {time.time() - t0:.0f} s", flush=True)


def pyramid_same(run, base_run):
    """does the pixel pyramid of the pair run equal the pyramid of the B-type run whose graph is delivered (the connectors then sit on the pixel features of that graph; `jends_eval.py score`'s check, repeated for the stack)"""
    import torch
    ck = torch.load(os.path.join(ROOT, "runs", run, "model.pt"), map_location="cpu")["state"]
    base = torch.load(os.path.join(ROOT, "runs", base_run, "model.pt"), map_location="cpu")["state"]
    keys = [k for k in base if k.startswith("pyr.") and (("jmodel." + k) in ck or k in ck)]
    _require(keys, f"no pyramid tensor of {base_run} is in {run}: the pyramid check would be empty")
    return all(torch.equal(ck["jmodel." + k] if "jmodel." + k in ck else ck[k], base[k]) for k in keys)


# ------------------------------------------------------------------------------------------------------------------------------ scoring
def pairs_of(e, thr, attach=False):
    """the pairs of the head at probability >= thr: (entry xy, entry heading, exit xy, exit heading[, the two handle scales]) as `qends_eval.cmd_score` (attach: the attached ends of the dump, E320)"""
    import qends_eval as QEV
    k = e["kp_pos"].shape[0] // 2
    p = 1.0 / (1.0 + np.exp(-e["logit"])) * e["allowed"]
    ia, ib = np.nonzero(p >= thr)
    return QEV.make_pairs(e, ia, ib, k, attach=attach)


def score_dumps(frames, graphs, ends, variants, thr=THR, n=0, log=print):
    """per frame: the counts of the delivered graph of every variant and the maneuver counts of those that have a label -> dict(counts={variant: [per-frame counts]}, man={variant: [(realised, maneuvers, hit, near)]}, curves, wired).
    graphs = {name: dump list}, ends = {name: {row: pair-dump frame}} (the sources the variants name); a dump list that two names share is read out once per frame"""
    import jmaneuver as JM
    import qends as QEND
    import qends_eval as QEV
    import qgraph_eval as QE
    import qgraph_readout as QR
    spec = QR.parse(CLOSE)
    cs = {v["name"]: [] for v in variants}
    man = {v["name"]: [] for v in variants if v["man"]}
    curves = {v["name"]: 0 for v in variants if v["ends"]}
    wired = dict(curves)
    t0 = time.time()
    for b, f in enumerate(frames):
        if n and b >= n:
            break
        read = {}
        for gname, dl in graphs.items():
            if id(dl) in read:
                read[gname] = read[id(dl)]
                continue
            d = dl[b]
            pred = QR.apply(dict(nodes=dict(pos=d["pos"], dirv=d["dirv"], e=d["e"], flags=d["flags"], section=d["section"], s=d.get("s")),
                                 links=dict(src=d["link_src"], dst=d["link_dst"], p=d["link_p"], long=d["link_long"], a=d["link_a"], b=d["link_b"])), spec)
            graph, bk = QE.readout(pred, 0.5, 0.5)
            read[gname] = read[id(dl)] = (graph, bk, np.asarray(pred["links"]["long"], bool))
        raws = {}
        for v in variants:
            graph, bk, long_mask = read[v["graph"]]
            if v["ends"] is None:
                raws[v["name"]] = QE.to_raw(graph)
                cs[v["name"]].append(QE.graph_counts(graph, f))
                continue
            pr = pairs_of(ends[v["ends"]][f.row], thr, v["attach"])
            raws[v["name"]], nc, nw = QEV.add_connectors(graph.lines, graph.edges, graph.nodes, bk["kind"], bk["piece_links"], long_mask, pr, snap=v["snap"], snap_nogeom=v["nogeom"])
            curves[v["name"]] += nc
            wired[v["name"]] += nw
            cs[v["name"]].append(QE.graph_counts(raws[v["name"]], f))
        if man:
            mans = JM.maneuvers(f.lane(), QEND.cluster)
            if mans:
                ref_tree = f.ref().tree
                for v in variants:
                    if v["man"]:
                        n_ok, nm, hit, near, _ = JM.frame_counts(raws[v["name"]], mans, ref_tree)
                        man[v["name"]].append((n_ok, nm, hit, near))
        f._lane = f._ref = None
        if (b + 1) % 100 == 0:
            log(f"  scored {b + 1}/{len(frames)}  {time.time() - t0:.0f} s")
    return dict(counts=cs, man=man, curves=curves, wired=wired)


def maneuver_numbers(man, variants):
    out = {}
    for v in variants:
        if v["man"]:
            x = np.array(man[v["name"]])
            out[v["man"]] = dict(maneuvers=int(x[:, 1].sum()), realised=int(x[:, 0].sum()), recall=float(x[:, 0].sum() / x[:, 1].sum()), frame_recall_mean=float(np.mean(x[:, 0] / x[:, 1])), frames_none=int((x[:, 0] == 0).sum()),
                                 frames_all=int((x[:, 0] == x[:, 1]).sum()), junction_precision=float(x[:, 2].sum() / max(1, x[:, 3].sum())), frames=len(x))
    return out


def paired(cs_a, cs_b, scenes, boot=1000, keys=KEYS):
    """winner minus control by the project's paired scene bootstrap (`autoresearch/Evaluation/stats.py`: a = variant, b = control; whole scenes drawn with replacement, the same draws for both)"""
    import qgraph_eval as QE
    from Evaluation import stats as ST
    f_a = lambda sel: QE.graph_numbers(cs_a if sel is None else [cs_a[i] for i in sel])  # noqa: E731
    f_b = lambda sel: QE.graph_numbers(cs_b if sel is None else [cs_b[i] for i in sel])  # noqa: E731
    return ST.paired_draws(f_a, f_b, ST.scene_draws(scenes, boot, 0), keys, better={k: "low" for k in LOWER})


# ------------------------------------------------------------------------------------------------------------------------------ cards: the variants, the control E312, the arms against it
def control_view(card):
    """E312's snapped read-out from either card layout: E321's `card_s<S>.json` (variants.snapped) or a stored VAL card of `qends_eval.py score` -> dict(numbers = the snapped head's numbers, plain = the plain graph's, curves, wired)"""
    c = card["variants"]["snapped"] if "variants" in card else card
    n = c["numbers"]
    return dict(numbers={k: float(n["head"][k]) for k in KEYS}, plain={k: float(n["base"][k]) for k in KEYS}, curves=c["curves"], wired=c["wired_both_ends"])


def guards_vs(num, ref):
    """the D79 guards of an arm against E312 (card E343 section 0): F1@1.0 loss <= 0.01, dup_frac rise <= 0.02, vertices per 100 m rise <= 1.0, junction F1 loss <= 0.03; the topo F1 gain is the reading's, not a guard. `dup_frac_e324_val_bound_ok`:
    E324's own VAL bound (rise <= 0.01), printed next to the D79 one and deciding nothing"""
    g = lambda k: float(num[k]) - float(ref[k])  # noqa: E731
    cond = dict(F1_1=g("F1@1.0") >= -0.01, dup_frac=g("dup_frac") <= 0.02, vert100_dp=g("vert100_dp") <= 1.0, junction_F1=g("junction_F1") >= -0.03)
    return dict(holds=bool(all(cond.values())), conditions={k: bool(v) for k, v in cond.items()}, dup_frac_rise=g("dup_frac"), dup_frac_e324_val_bound_ok=bool(g("dup_frac") <= 0.01))


def match(new, old, tol, path=""):
    """(ok, worst abs difference, where) of the numbers of `old` (nested dicts) against `new`: an integer within the relative `count_rel`, topo F1 within `topo`, any other real within `real`; a key of `old` that `new` lacks fails"""
    ok, worst, where = True, 0.0, ""
    for k, v in old.items():
        p = f"{path}/{k}"
        if k not in new:
            return False, float("inf"), p + " (missing)"
        if isinstance(v, dict):
            o, w, wh = match(new[k], v, tol, p)
        elif isinstance(v, bool) or not isinstance(v, (int, float)):
            continue
        else:
            d = abs(float(new[k]) - float(v))
            o, w, wh = (d <= tol["count_rel"] * abs(v) if isinstance(v, int) else d <= tol["topo" if k == "topo_F1" else "real"]), (d if d == d else float("inf")), p
        ok = ok and o
        if w > worst:
            worst, where = w, wh
    return ok, worst, where


def counts_maxdiff(a, b):
    """max abs difference of two per-frame count structures (dicts, lists, arrays, numbers); inf when their structure differs"""
    if isinstance(a, dict):
        return float("inf") if not isinstance(b, dict) or a.keys() != b.keys() else max([0.0] + [counts_maxdiff(a[k], b[k]) for k in a])
    if isinstance(a, (list, tuple)):
        return float("inf") if not isinstance(b, (list, tuple)) or len(a) != len(b) else max([0.0] + [counts_maxdiff(x, y) for x, y in zip(a, b)])
    if a is None or b is None:
        return 0.0 if a is b else float("inf")
    try:
        x, y = np.asarray(a, float), np.asarray(b, float)
    except (TypeError, ValueError):
        return 0.0 if a == b else float("inf")
    if x.shape != y.shape:
        return float("inf")
    d = np.where(np.isnan(x) & np.isnan(y), 0.0, np.abs(x - y))
    return float("inf") if np.isnan(d).any() else float(d.max(initial=0.0))


def make_card(seed, run, split, frames, sc, boot, setup, variants, control=None, use_counts=True):
    """the card of one seed: per variant that has a base the numbers, the difference to the base, the D79 rule of its seed and the paired interval (as the E312 cards); the maneuver coverage; for a setup with arms and a `control` (E312's stored numbers +
    per-frame counts when they reproduced) the arms against E312, k - w and w - p"""
    import qgraph_eval as QE
    from qends_eval import holds
    names = [v["name"] for v in variants]
    num = {n_: QE.graph_numbers(sc["counts"][n_]) for n_ in names}
    card = dict(split=split, run=run, seed=seed, frames=len(frames), scenes=int(len(set(f.scene for f in frames))), thr=THR, close_step=CLOSE, date=time.strftime("%Y-%m-%d %H:%M"), variants={})
    scored = [v for v in variants if v["base"]]
    for v in scored:
        ok, cond = holds(num[v["name"]], num[v["base"]])
        card["variants"][v["name"]] = dict(base=v["base"], curves=sc["curves"][v["name"]], wired_both_ends=sc["wired"][v["name"]], numbers=dict(base={k: float(num[v["base"]][k]) for k in KEYS}, head={k: float(num[v["name"]][k]) for k in KEYS}),
                                          difference={k: float(num[v["name"]][k] - num[v["base"]][k]) for k in KEYS}, rule_d79_this_seed=dict(holds=bool(ok), conditions={k: bool(x) for k, x in cond.items()}))
    card["maneuver"] = maneuver_numbers(sc["man"], variants)
    scenes = [f.scene for f in frames][: len(sc["counts"]["plain"])]
    if boot:
        card["paired"] = {v["name"]: paired(sc["counts"][v["name"]], sc["counts"][v["base"]], scenes, boot) for v in scored}
        card["paired_note"] = f"variant minus its plain delivered graph (B or c2 + the close-node step), {boot} scene draws of {len(set(scenes))} scenes, same draws for both (autoresearch/Evaluation/stats.py)"
    if control is not None:
        arms = {a_: v_ for a_, v_ in setup["arms"].items() if v_ in num}
        cmp_ = dict(control=dict(card=control["card"], numbers=control["numbers"], plain_check=control["plain_check"], repro=control["repro"], per_frame_counts_used=bool(use_counts and control["counts"] is not None)), vs_e312={})
        for a_, v_ in arms.items():
            e = dict(read_out=v_, difference={k: float(num[v_][k]) - control["numbers"][k] for k in KEYS}, guards=guards_vs(num[v_], control["numbers"]))
            if boot and use_counts and control["counts"] is not None:
                e["paired"] = paired(sc["counts"][v_], control["counts"], scenes, boot)
            cmp_["vs_e312"][a_] = e
        if boot and "k" in arms:
            cmp_["k_minus_w"] = paired(sc["counts"][arms["k"]], sc["counts"][arms["w"]], scenes, boot)
        if boot and "p" in arms:
            cmp_["w_minus_p"] = paired(sc["counts"][arms["w"]], sc["counts"][arms["p"]], scenes, boot)
        card["compare"] = cmp_
    return card


# ------------------------------------------------------------------------------------------------------------------------------ the reading (card E343 section 0): the label is code, nobody reads the rule by hand
FULL_D, KEEP_D, LO_FLOOR = 0.010, 0.005, -0.005


def reading(seeds):
    """the label of the recommendation, from UNROUNDED numbers of arm w against E312, one dict per seed: D (topo F1 difference), lo / hi (95 % paired scene interval of D; None / nan when there is none), guards (the D79 guards against E312 hold).
    FULL RETENTION: D >= 0.010, lo > 0 on every seed and the guards hold; NOT CONTRADICTED: not full, D > 0, lo > -0.005 on every seed and the guards hold; CONTRADICTED otherwise (a guard miss on a seed is contradicted).
    No interval on one seed (the per-frame counts of E312 did not reproduce, or on one seed only) means no interval on any: the fallback, labelled 'without interval': full = D >= 0.010, not contradicted = D >= 0.005 on every seed, guards as above"""
    assert seeds, "reading() needs the seeds"
    has = lambda x: x is not None and x == x  # noqa: E731
    guards = all(s["guards"] for s in seeds)
    if all(has(s.get("lo")) for s in seeds):
        if guards and all(s["D"] >= FULL_D and s["lo"] > 0 for s in seeds):
            return "FULL RETENTION"
        if guards and all(s["D"] > 0 and s["lo"] > LO_FLOOR for s in seeds):
            return "NOT CONTRADICTED"
        return "CONTRADICTED"
    if guards and all(s["D"] >= FULL_D for s in seeds):
        return "FULL RETENTION (without interval)"
    if guards and all(s["D"] >= KEEP_D for s in seeds):
        return "NOT CONTRADICTED (without interval)"
    return "CONTRADICTED (without interval)"


def reading_inputs(card):
    """the numbers `reading` takes from the card of one seed (arm w against E312)"""
    e = card["compare"]["vs_e312"]["w"]
    t = e.get("paired", {}).get("topo_F1", {})
    return dict(D=e["difference"]["topo_F1"], lo=t.get("lo"), hi=t.get("hi"), guards=e["guards"]["holds"])


RECOMMENDATION = {"FULL RETENTION": "E324 s may become the system of record in PROBLEM.md and the registry (Mirko's decision, with the D94 / D95 grants)",
                  "NOT CONTRADICTED": "VAL gain not confirmed on TEST; the promotion rests on VAL (Mirko decides)",
                  "CONTRADICTED": "no edit; E312's TEST numbers stay the reference"}


# ------------------------------------------------------------------------------------------------------------------------------ report
def print_card(card):
    print(f"{card['split']} seed {card['seed']} run {card['run']}: {card['frames']} frames, {card['scenes']} scenes")
    print(f"{'':20s}" + "".join(f"{k:>12s}" for k in KEYS))
    shown = set()
    for v, c in card["variants"].items():
        if c["base"] not in shown:
            shown.add(c["base"])
            print(f"{c['base'] + ' graph':20s}" + "".join(f"{c['numbers']['base'][k]:12.4f}" for k in KEYS))
        print(f"{v:20s}" + "".join(f"{c['numbers']['head'][k]:12.4f}" for k in KEYS))
        print(f"{'  - ' + c['base']:20s}" + "".join(f"{c['difference'][k]:+12.4f}" for k in KEYS))
    for v, m in card["maneuver"].items():
        print(f"maneuvers {v:9s} realised {m['realised']}/{m['maneuvers']} = {m['recall']:.3f}; junction precision {m['junction_precision']:.3f}")
    ci = lambda p, k: f"{k} {p[k]['diff']:+.4f} [{p[k]['lo']:+.4f}, {p[k]['hi']:+.4f}]"  # noqa: E731
    for v, p in card.get("paired", {}).items():
        print(f"paired {v} - {card['variants'][v]['base']}: " + ", ".join(ci(p, k) for k in ("topo_F1", "lanes_found", "junction_F1", "dup_frac") if k in p))
    cmp_ = card.get("compare")
    if cmp_:
        c = cmp_["control"]
        print(f"control E312 ({c['card']}): plain graph max abs difference {c['plain_check']['max_abs_diff']:.2e} (tolerance {c['plain_check']['tolerance']}); per-frame counts " + (f"reproduce to {c['repro']['max_abs_diff']:.2e}" if c["repro"] else "not re-scored"))
        for a_, e in cmp_["vs_e312"].items():
            print(f"arm {a_} ({e['read_out']}) - E312: topo F1 {e['difference']['topo_F1']:+.4f}" + (f" [{e['paired']['topo_F1']['lo']:+.4f}, {e['paired']['topo_F1']['hi']:+.4f}]" if "paired" in e else " (no interval)")
                  + f"; D79 guards against E312 {'hold' if e['guards']['holds'] else 'MISS'} {e['guards']['conditions']}")
        for key, text in (("k_minus_w", "k - w"), ("w_minus_p", "w - p")):
            if key in cmp_:
                print(f"{text}: " + ci(cmp_[key], "topo_F1") + ", " + ci(cmp_[key], "lanes_found"))


def results_md(cards, split, setup):
    """results.md of the cards of one setup, all numbers from the cards; for E324 s the reading of the card E343 (label by `reading`)"""
    c0 = cards[0]
    ci = lambda d, p, k: f"{d[k]:+.4f}" + (f" [{p[k]['lo']:+.4f}, {p[k]['hi']:+.4f}]" if k in p else "")  # noqa: E731
    if setup["arms"]:
        L = [f"# E343 - {setup['label']} on {split.upper()} ({c0['frames']} frames, {c0['scenes']} scenes): the arms against E312", "",
             f"Report only: nothing selected, tuned or re-read on this split. Control c = E312 snapped ({'the stored card of E321' if split == 'test' else 'its stored VAL card'}); w = E324 s snapped; p = the PRIMARY end-to-end read-out (attached ends + snap-nogeom wiring); k = the stack E338 (E324 s's pair head on the delivered graph of c2, snapped); "
             "snap_nogeom is reported only. The reading below is the rule of the card (`reading()`), no promotion of p or k is recommended whatever the numbers.", ""]
        seeds = [reading_inputs(c) for c in cards]
        label = reading(seeds) if len(cards) == 2 else "not applied (the rule needs both seeds)"
        L += ["## reading (card E343 section 0)", "", f"**{label}**", "", "| seed | D = topo F1 w - E312 | lo | hi | D79 guards against E312 | dup_frac rise (D79 bound 0.02 decides; E324 VAL bound 0.01 printed only) |", "|---|---|---|---|---|---|"]
        for c, s in zip(cards, seeds):
            g = c["compare"]["vs_e312"]["w"]["guards"]
            L.append(f"| {c['seed']} | {s['D']:+.6f} | " + (f"{s['lo']:+.6f} | {s['hi']:+.6f}" if s["lo"] is not None else "- | -") + f" | {'hold' if g['holds'] else 'MISS ' + str({k: v for k, v in g['conditions'].items() if not v})} | {g['dup_frac_rise']:+.4f} ({'ok' if g['conditions']['dup_frac'] else 'miss'} / {'ok' if g['dup_frac_e324_val_bound_ok'] else 'miss'}) |")
        L += ["", "Recommendation: " + RECOMMENDATION.get(label.split(" (")[0], "none (the rule needs both seeds)"), ""]
        L += ["## every arm against E312 snapped (D = arm - E312, paired 95 % scene bootstrap when E312's per-frame counts reproduced)", "",
              "| arm | seed | topo F1 E312 | topo F1 arm | D | lanes found | junction F1 | dup_frac | F1@1.0 | vertices / 100 m | D79 guards |", "|---|---|---|---|---|---|---|---|---|---|---|"]
        for a_ in setup["arms"]:
            for c in cards:
                e = c["compare"]["vs_e312"].get(a_)
                if e:
                    p = e.get("paired", {})
                    ref = c["compare"]["control"]["numbers"]["topo_F1"]
                    L.append(f"| {a_} ({e['read_out']}) | {c['seed']} | {ref:.4f} | {ref + e['difference']['topo_F1']:.4f} | {ci(e['difference'], p, 'topo_F1')} | {ci(e['difference'], p, 'lanes_found')} | {ci(e['difference'], p, 'junction_F1')} | {ci(e['difference'], p, 'dup_frac')} | "
                             f"{ci(e['difference'], p, 'F1@1.0')} | {ci(e['difference'], p, 'vert100_dp')} | {'hold' if e['guards']['holds'] else 'MISS'} |")
        L += ["", "## arms against each other (topo F1 difference, paired 95 % scene bootstrap)", "", "| comparison | seed | topo F1 | lanes found | junction F1 | dup_frac |", "|---|---|---|---|---|---|"]
        for key, text in (("k_minus_w", "k - w (stack - snapped)"), ("w_minus_p", "w - p (snapped - primary: the measured price of the end-to-end read-out)")):
            for c in cards:
                p = c["compare"].get(key)
                if p:
                    L.append(f"| {text} | {c['seed']} | " + " | ".join(f"{p[k]['diff']:+.4f} [{p[k]['lo']:+.4f}, {p[k]['hi']:+.4f}]" for k in ("topo_F1", "lanes_found", "junction_F1", "dup_frac")) + " |")
        L += ["", "E2E gap (named): arm w contains a rule step (the snap's wiring and keypoint geometry); arm p is the end-to-end read-out.", ""]
    else:
        L = [f"# {setup['label']} on {split.upper()} ({c0['frames']} frames, {c0['scenes']} scenes) - winner against the plain delivered graph of B", "",
             "Read-outs fixed on VAL: connectors at p >= 0.1 snapped (E311, the card of record of E312) and not snapped (the differentiable read-out); control = B + the close-node step `close:1.5@0.5`. Report only: nothing selected or tuned on this split.", ""]
    names = [v for v in c0["variants"]]
    for v in names:
        L += [f"## {v} against its plain delivered graph ({c0['variants'][v]['base']}; connectors at p >= {THR})", "",
              "| seed | topo F1 plain | topo F1 | diff [95 % scene bootstrap] | lanes found diff | junction F1 diff | dup_frac diff | F1@1.0 diff | vertices / 100 m diff |", "|---|---|---|---|---|---|---|---|---|"]
        for c in cards:
            d, h, p = c["variants"][v]["difference"], c["variants"][v]["numbers"], c.get("paired", {}).get(v, {})
            L.append(f"| {c['seed']} | {h['base']['topo_F1']:.4f} | {h['head']['topo_F1']:.4f} | {ci(d, p, 'topo_F1')} | {ci(d, p, 'lanes_found')} | {ci(d, p, 'junction_F1')} | {ci(d, p, 'dup_frac')} | {ci(d, p, 'F1@1.0')} | {ci(d, p, 'vert100_dp')} |")
        L.append("")
    labels = list(c0["maneuver"])
    L += ["## maneuver coverage (E310)" + ("" if setup["stored_man"] else " - not verified against a stored card (none exists); its VAL value is a plausibility line"), "", "| seed | " + " | ".join(labels) + " |", "|---|" + "---|" * len(labels)]
    for c in cards:
        L.append(f"| {c['seed']} | " + " | ".join(f"{100 * c['maneuver'][x]['recall']:.1f} %" for x in labels) + " |")
    if c0.get("device"):
        drift = c0.get("code_drift")
        L += ["", "## provenance", "", f"Device of the dumps and the scoring: {c0['device']}. Imported lane code: {len(c0['code_sha256'])} files, digest {c0['code_digest'][:16]}; " + (f"CODE DRIFT during the run: {', '.join(drift)}." if drift else "no change of those files during the run.")]
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------------------------------------------------------------------ the commands
def sub_seed(p, seed):
    return p.replace("{seed}", str(seed)) if p else p


def seed_inputs(a, setup, seed, stack, control_ends_flag):
    """the paths of one seed: the dumps (own / B / c2) and the control's (per split: CONTROL_DEFAULTS, a flag wins); a path named by a flag must exist, a default path is reused when it exists and made otherwise (never the control's, and never B's on TEST: E321's B dump is reused)"""
    d, run = dump_dir(a.split, a.max), f"{a.run_prefix}{seed}"
    cd = CONTROL_DEFAULTS[a.split]
    b_flag = sub_seed(a.b_dump, seed)
    i = dict(run=run, b=b_flag or os.path.join(d, f"predfull_{sub_seed(setup['graph_run'], seed)}.pkl"), b_made=not b_flag and a.split != "test", ends=os.path.join(d, f"ends_{run}.pkl"),
             c2=os.path.join(d, f"predfull_{sub_seed(setup['stack_run'], seed)}.pkl") if stack else None, card=os.path.join(ROOT, sub_seed(a.control_card or cd["card"], seed)))
    ctl_ends = sub_seed(a.control_ends_dump, seed) or (os.path.join(ROOT, sub_seed(cd["ends"], seed)) if cd["ends"] else None)
    i["ctl_ends"] = ctl_ends
    i["ctl_b"] = os.path.join(ROOT, sub_seed(cd["b"], seed)) if cd["b"] and not control_ends_flag else i["b"]
    return i


def tol_of(split):
    """the tolerance of the plain-graph check against the control card: TEST reads E321's B dump (equal to 1e-9), a VAL rehearsal makes a fresh CPU dump (the tolerance of criterion 2)"""
    return EXACT if split == "test" else REHEARSAL


def numbers_of(counts):
    import qgraph_eval as QE
    n = QE.graph_numbers(counts)
    return {k: float(n[k]) for k in KEYS}


def check_seed(run, seed):
    """the pair run is the run of this seed (its keypoints sit on that seed's graph): its `config.json` says so"""
    with open(os.path.join(ROOT, "runs", run, "config.json")) as f:
        ck = json.load(f)
    if ck.get("seed", seed) != seed:
        sys.exit(f"the run {run} has seed {ck.get('seed')}, not {seed}: its keypoints would sit on another seed's graph")


def precheck(a, setup, seed, stack):
    """everything that can be refused before hours of dumping: the seed of the pair run, the pixel pyramid of the stack (arm k fails its own check: run again with --no-stack), the dumps that a flag or the split's defaults name, the control's card
    -> (the paths of the seed, E312's card or None)"""
    i = seed_inputs(a, setup, seed, stack, bool(a.control_ends_dump))
    check_seed(i["run"], seed)
    if stack and not pyramid_same(i["run"], sub_seed(setup["stack_run"], seed)):
        sys.exit(f"STOP (seed {seed}): the pixel pyramid of {i['run']} is not that of {sub_seed(setup['stack_run'], seed)}, so the stack's connectors would sit on other pixel features than the delivered graph: arm k fails its check, run again with --no-stack")
    named = ([] if i["b_made"] else [i["b"]]) + ([i["ctl_ends"], i["ctl_b"]] if setup["arms"] and i["ctl_ends"] else []) + ([i["card"]] if setup["arms"] else [])
    missing = [p for p in named if not os.path.exists(p)]
    if missing:
        sys.exit(f"STOP (seed {seed}): not there, and named by a flag or by the split's defaults (B's dump is never made on TEST): {missing}")
    ctl = None
    if setup["arms"]:
        with open(i["card"]) as f:
            ctl = json.load(f)
    return i, ctl


def score_seed(a, setup, seed, frames, stack, pre):
    """one seed (`pre`: what `precheck` returned): the dumps (validated; one that this module made earlier is reused only under the same provenance), the scoring of every variant, the control's checks
    -> dict(seed, run, sc, variants, control); stops before any number is printed when the plain graph differs from the control card's"""
    i, ctl_card = pre
    n, split = a.max, a.split
    run_b, run_c = sub_seed(setup["graph_run"], seed), sub_seed(setup.get("stack_run"), seed)
    exp = lambda run: provenance(run, a.device)  # noqa: E731
    dumps = {"b": get_dump(i["b"], (lambda: dump_b(split, run_b, i["b"], a.device, n)) if i["b_made"] else None, split, "B's dump", n, exp(run_b) if i["b_made"] else None)}
    ends = {"own": {f["row"]: f for f in get_dump(i["ends"], lambda: dump_ends(split, i["run"], i["ends"], a.device, n), split, f"pair dump of {i['run']}", n, exp(i["run"]))}}
    if stack:
        dumps["c2"] = get_dump(i["c2"], lambda: dump_b(split, run_c, i["c2"], a.device, n), split, "c2's dump", n, exp(run_c))
    control = view = None
    if setup["arms"]:
        view = control_view(ctl_card)
        control = dict(card=os.path.relpath(i["card"], ROOT).replace(os.sep, "/"), numbers=view["numbers"], counts=None, repro=None)
        if i["ctl_ends"]:
            dumps["ctl"] = dumps["b"] if i["ctl_b"] == i["b"] else load_dump(i["ctl_b"], split, "the control's B dump", n)
            ends["ctl"] = {f["row"]: f for f in load_dump(i["ctl_ends"], split, "the control's pair dump", n)}
    variants = variants_of(setup, stack, control=bool(control) and "ctl" in ends)
    sc = score_dumps(frames, dumps, ends, variants, n=n)
    if control:
        tol = tol_of(split)
        ok, worst, where = match(numbers_of(sc["counts"]["plain"]), view["plain"], tol)
        control["plain_check"] = dict(max_abs_diff=worst, where=where, tolerance=f"topo {tol['topo']:g}, others {tol['real']:g}", ok=bool(ok or n))                  # a smoke run (the first n frames) cannot reproduce a whole card
        if not (ok or n):
            sys.exit(f"KILL (seed {seed}): the plain graph computed from B's dump differs from the control card's numbers.base by {worst:.2e} ({where}): no number of any arm is used")
        if "ctl" in ends:
            r_ok, r_worst, r_where = match(numbers_of(sc["counts"]["e312_snapped"]), view["numbers"], EXACT)
            wiring = (sc["curves"]["e312_snapped"], sc["wired"]["e312_snapped"]) == (view["curves"], view["wired"])                                                 # E312's curves and wired counts too
            control["repro"] = dict(max_abs_diff=r_worst, where=r_where, ok=bool(r_ok and wiring), curves=[sc["curves"]["e312_snapped"], view["curves"]], wired=[sc["wired"]["e312_snapped"], view["wired"]])
            control["counts"] = sc["counts"]["e312_snapped"] if control["repro"]["ok"] else None
    return dict(seed=seed, run=i["run"], sc=sc, variants=variants, control=control)


def check_stored(setup, run, seed, card, sc, tol, frames=None):
    """the stored VAL cards of the setup against the card just made: the numbers (head, plain, difference), curves and wired counts within `tol`, the stored spec fields equal to the variant's; with EXACT also the per-frame counts equal to the stored
    `.counts.pkl` and the scenes equal (a card of `setup["stored_counts"]` without that file fails) -> (ok, lines, failed): `failed` names the read-outs whose stored card did not reproduce. Only the read-outs in `card` are checked: without the stack
    (`--no-stack`) the stack card is not looked at and w and p are checked alone"""
    lines, failed = [], []
    by_name = {v["name"]: v for v in setup["variants"] + setup["stack"]}
    for name, fname in setup["stored"].items():
        if name not in card["variants"]:
            continue
        path = os.path.join(ROOT, "runs", run, fname)
        old, new, v = json.load(open(path)), card["variants"][name], by_name[name]
        want = dict(thr=THR, spec=CLOSE, snap=v["snap"], attach=v["attach"], attach_r=None, snap_options=["snap_nogeom"] if v["nogeom"] else [], wire=None)
        if tol is EXACT:                                                                              # the stored dumps made the stored cards; a fresh dump has another path
            want["full"] = sub_seed(STORED_DUMPS[v["graph"]], seed)
        wrong = {k: (old[k], w) for k, w in want.items() if k in old and old[k] != w}
        o1, w1, wh1 = match(new["numbers"], old["numbers"], tol)
        o2, w2, wh2 = match(new["difference"], old["difference"], tol)
        o3, w3, wh3 = match(dict(curves=new["curves"], wired=new["wired_both_ends"]), dict(curves=old["curves"], wired=old["wired_both_ends"]), tol)
        line = f"CHECK {name}: max abs difference of the numbers {w1:.2e} ({wh1}), of the differences {w2:.2e}; curves {new['curves']} / {old['curves']}, wired {new['wired_both_ends']} / {old['wired_both_ends']} ({'within' if o3 else 'OUT'})"
        good = o1 and o2 and o3 and not wrong
        if wrong:
            line += f"; STORED SPEC DIFFERS {wrong}"
        cpath = path[:-5] + ".counts.pkl"
        if tol is EXACT and os.path.exists(cpath):
            with open(cpath, "rb") as f:
                st = pickle.load(f)
            dh, db = counts_maxdiff(sc["counts"][name], st["head"]), counts_maxdiff(sc["counts"][v["base"]], st["base"])
            same = st["scenes"] == [f.scene for f in frames] and len(st["head"]) == len(sc["counts"][name])
            line += f"; per-frame counts: max abs difference {max(dh, db):.2e} (head {dh:.2e}, plain {db:.2e}) over {len(st['head'])} frames, scenes {'equal' if same else 'DIFFER'}"
            good = good and dh == 0.0 and db == 0.0 and same
        elif tol is EXACT and name in setup["stored_counts"]:
            line += "; per-frame counts: the stored .counts.pkl is MISSING"
            good = False
        elif tol is EXACT:
            line += "; per-frame counts: no stored file"
        if not good:
            failed.append(name)
        lines.append(line + ("" if good else "  <-- DIFFERENCE"))
    if setup["stored_man"]:
        with open(os.path.join(ROOT, "runs", run, setup["stored_man"])) as f:
            old = json.load(f)["variants"]
        for lab, new in card["maneuver"].items():
            o, w, wh = match(new, old[lab], tol)
            if not o:
                failed.append(f"maneuver {lab}")
            lines.append(f"CHECK maneuver {lab}: realised {new['realised']} / {old[lab]['realised']} of {new['maneuvers']} / {old[lab]['maneuvers']}, max abs difference {w:.2e} ({wh})" + ("" if o else "  <-- DIFFERENCE"))
    else:
        lines.append("maneuver coverage: not verified against a stored card (none exists for this setup): " + ", ".join(f"{k} {100 * m['recall']:.1f} %" for k, m in card["maneuver"].items()))
    return not failed, lines, failed


def arm_k_only(failed):
    """does ONLY the stack (arm k) fail its check? Then the card's rule applies: run again with --no-stack, w and p are checked alone (decided from VAL only)"""
    return bool(failed) and set(failed) <= {"stack"}


def counts_reproduced(done):
    """E312's per-frame counts give the paired intervals only when they reproduced the stored card on EVERY seed (else no interval on any seed: the fallback of `reading`)"""
    return all(d["control"] and d["control"]["counts"] is not None for d in done)


def cmd_all(a):
    require_gate(a)
    setup = SETUPS.get(a.run_prefix) or sys.exit(f"--run-prefix {a.run_prefix!r}: one of {', '.join(SETUPS)}")
    stack = bool(setup["stack"]) and not a.no_stack
    out = out_dir(setup, a.split, a.max)
    for path in (out, dump_dir(a.split, a.max)):                                                     # before hours of work: E321's folders cannot be written (E312 on TEST is E321's, already done)
        if is_protected(path):
            sys.exit(f"refused: {path} lies under {' / '.join(PROTECTED)}")
    targets = [os.path.join(out, f"card_s{s}.json") for s in a.seeds] + [os.path.join(out, "results.md")]
    old = [p for p in targets if os.path.exists(p)]
    if old and not a.max:
        sys.exit(f"refused: {old} exist: the cards and results.md are written once (move them away to run again)")
    guard_e321("before the run")
    pre = {s: precheck(a, setup, s, stack) for s in a.seeds}                                         # seeds, pyramid, flags, control card: refused here, not after hours of dumping
    frames = frames_of(a.split)[: a.max or None]
    code0, done = code_sha256(), []
    for s in a.seeds:
        t0 = time.time()
        done.append(score_seed(a, setup, s, frames, stack, pre[s]))
        print(f"seed {s} scored in {time.time() - t0:.0f} s", flush=True)
    use_counts = counts_reproduced(done)
    if setup["arms"] and not use_counts:
        print("the per-frame counts of E312 are not used or did not reproduce on every seed: NO interval against E312 on any seed (the fallback of reading())", flush=True)
    rehearsal = a.split == "val" and not a.max
    cards, ok, failed = [], True, []
    for d in done:
        card = make_card(d["seed"], d["run"], a.split, frames[: len(d["sc"]["counts"]["plain"])], d["sc"], a.boot, setup, d["variants"], d["control"], use_counts)
        if rehearsal:                                                                                # the dress rehearsal: the fresh dumps against the stored VAL cards, the tolerance of criterion 2
            good, lines, bad = check_stored(setup, d["run"], d["seed"], card, d["sc"], REHEARSAL)
            card["rehearsal_check"] = dict(ok=bool(good), lines=lines, failed=bad)
            ok, failed = ok and good, failed + bad
        cards.append(card)
    code1 = code_sha256()                                                                            # the imported lane code at the end: a file that changed during the run is named (not fatal)
    drift = []
    for c in cards:
        c.update(device=a.device, code_sha256=code0, code_digest=code_digest(code0), code_drift=drift)
    if drift:
        print("CODE DRIFT:", ", ".join(drift), flush=True)
    for c in cards:
        print_card(c)
        for line in c.get("rehearsal_check", {}).get("lines", []):
            print(line)
    for c in cards:
        save_json(os.path.join(out, f"card_s{c['seed']}.json"), c, overwrite=bool(a.max))
    save_text(os.path.join(out, "results.md"), results_md(cards, a.split, setup), overwrite=bool(a.max))
    if setup["arms"] and len(cards) == 2:
        print("READING:", reading([reading_inputs(c) for c in cards]))
    guard_e321("after the run")
    if rehearsal:
        print("REHEARSAL CHECK:", "PASS" if ok else "FAIL (a tolerance of criterion 2 is exceeded)")
        if stack and arm_k_only(failed):
            print("only arm k fails its check: run again with --no-stack (w and p are then checked alone)")
    return 0 if ok else 1


def cmd_verify(a):
    """VAL, from the stored dumps: the scorer of this module against the stored cards of the setup (criterion 1 of E343: every number within 1e-9, curves and wired counts equal, per-frame counts equal to the stored `.counts.pkl`). Without `--no-stack` the stack
    (arm k) is checked too and its failure makes the exit code non-zero; with it w and p are checked alone"""
    run = a.run or f"e312_p_s{a.seed}"
    setup, seed = setup_of(run)
    stack = bool(setup["stack"]) and not a.no_stack
    variants = variants_of(setup, stack)
    frames = frames_of("val")
    dumps = {"b": load_dump(os.path.join(ROOT, sub_seed(STORED_DUMPS["b"], seed)), "val", "the stored B dump")}
    if stack:
        dumps["c2"] = load_dump(os.path.join(ROOT, sub_seed(STORED_DUMPS["c2"], seed)), "val", "the stored c2 dump")
    ends = {"own": {f["row"]: f for f in load_dump(os.path.join(ROOT, "runs", run, "ends_val.pkl"), "val", f"the stored pair dump of {run}")}}
    check_seed(run, seed)
    sc = score_dumps(frames, dumps, ends, variants)
    card = make_card(seed, run, "val", frames, sc, a.boot, setup, variants)
    save_json(os.path.join(dump_dir("val"), "val_check", f"verify_card_{run}.json"), card)                # for E312 in the layout of E321's cards: the control card of a rehearsal
    print_card(card)
    ok, lines, failed = check_stored(setup, run, seed, card, sc, EXACT, frames)
    for line in lines:
        print("VERIFY " + line)
    if stack:                                                                                          # the stored stack card was made under this check of `jends_eval.py score --base`
        same = pyramid_same(run, sub_seed(setup["stack_run"], seed))
        print(f"VERIFY pyramid of {run} == the pyramid of {sub_seed(setup['stack_run'], seed)}: {same}")
        if not same:
            ok, failed = False, failed + ["stack"]
    print("VERIFY RESULT:", f"all reproduced ({run}: numbers to 1e-9, curves and wired counts equal, per-frame counts equal)" if ok else "DIFFERENCES (see above)")
    if stack and arm_k_only(failed):
        print("VERIFY only arm k fails its check: run again with --no-stack (w and p are then checked alone)")
    return 0 if ok else 1



# ------------------------------------------------------------------------------------------------------------------------------ self-test
def selftest():
    # (1) the gate: refuses TEST without an id, outside the launcher, with an id the log does not open, with --max; opens only for a logged id under the launcher
    log = "| date | loop | what | who / which run | numbers | note |\n|---|---|---|---|---|---|\n| d | 1 | design | run id E9.ok | - | Mirko's OK, once |\n| d | 1 | design | run id E9.no | - | pending |\n"
    assert not test_gate_open("", "", log, set())[0]
    assert not test_gate_open("E9.ok", "", log, set())[0], "outside the launcher"
    assert not test_gate_open("E9.ok", "E9.other", log, set())[0], "AR_RUN_ID of another run"
    assert not test_gate_open("E9.no", "E9.no", log, set())[0], "a row without the OK"
    assert not test_gate_open("E9.none", "E9.none", log, set())[0], "an id the log does not name"
    assert test_gate_open("E9.ok", "E9.ok", log, set())[0]
    a = argparse.Namespace(split="test", test_id="E9.ok", max=0)
    old, open_gate = os.environ.pop("AR_RUN_ID", None), test_gate_open
    try:
        for max_, why in ((0, "without the launcher"), (8, "with --max, although the gate is open")):
            a.max = max_
            globals()["test_gate_open"] = (lambda *x, **k: (True, "")) if max_ else open_gate               # the --max refusal is the only one that can stop the second case
            try:
                require_gate(a)
                raise AssertionError(f"require_gate let TEST through {why}")
            except SystemExit:
                pass
        a.split = "val"
        require_gate(a)                                                                            # VAL needs no gate
    finally:
        globals()["test_gate_open"] = open_gate
        os.environ.pop("AR_RUN_ID", None)
        if old is not None:
            os.environ["AR_RUN_ID"] = old
    # (2) the splits: ids only
    r_val, r_test = split_rows("val"), split_rows("test")
    assert len(r_val) == EXPECT_FRAMES["val"] and len(r_test) == EXPECT_FRAMES["test"] and not set(r_val.tolist()) & set(r_test.tolist())
    # (3) the paired bootstrap on toy counts (graph_numbers replaced by a mean of one field): a variant that is 0.1 better in every frame gives an interval around +0.1, a system against itself exactly 0
    import qgraph_eval as QE
    saved = QE.graph_numbers
    QE.graph_numbers = lambda cs: {"topo_F1": float(np.mean([c["x"] for c in cs])), "dup_frac": float(np.mean([c["d"] for c in cs]))}
    try:
        rng = np.random.RandomState(1)
        base_c = [dict(x=float(v), d=0.2) for v in rng.rand(40)]
        var_c = [dict(x=c["x"] + 0.1, d=0.2) for c in base_c]
        scenes = [i // 5 for i in range(40)]
        p = paired(var_c, base_c, scenes, 200, keys=("topo_F1", "dup_frac"))
        assert abs(p["topo_F1"]["diff"] - 0.1) < 1e-9 and abs(p["topo_F1"]["lo"] - 0.1) < 1e-9 and abs(p["topo_F1"]["hi"] - 0.1) < 1e-9, p["topo_F1"]
        p0 = paired(base_c, base_c, scenes, 200, keys=("topo_F1",))
        assert p0["topo_F1"]["diff"] == 0.0 and p0["topo_F1"]["lo"] == 0.0 and p0["topo_F1"]["hi"] == 0.0
    finally:
        QE.graph_numbers = saved
    # (4) the frames of the split val are exactly those of the scorers (`qgraph_eval.val_frames()`); the test branch is the same code on the test rows
    fv, fo = frames_of("val"), QE.val_frames()
    assert [(f.k, f.row, f.token, f.scene) for f in fv] == [(f.k, f.row, f.token, f.scene) for f in fo] and all(len(x.gt["pts"]) == len(y.gt["pts"]) for x, y in zip(fv, fo))
    # (5) the results text of the E312 setup
    num = lambda x: {k: x for k in KEYS}  # noqa: E731
    mk = lambda seed, h: dict(split="val", seed=seed, frames=3, scenes=2, run="r", maneuver={"plain": dict(recall=0.5), "p0.1": dict(recall=0.7), "p0.1s": dict(recall=0.71)}, paired={},
                              variants={v: dict(base="plain", difference=num(0.01), numbers=dict(base=num(0.5), head=num(0.51))) for v in ("unsnapped", "snapped")})  # noqa: E731
    txt = results_md([mk(0, 0), mk(1, 0)], "val", SETUPS["e312_p_s"])
    assert "snapped" in txt and "maneuver coverage" in txt and "stored card" not in txt
    # (6) the variants: every base exists in its setup, the labels are unique, the arms are variants of the setup
    for st in SETUPS.values():
        vs = variants_of(st, True, True)
        names = [v["name"] for v in vs]
        assert len(set(names)) == len(names) and all(v["base"] in names for v in vs if v["base"]) and all(v in names for v in st["arms"].values()) and set(st["stored"]) <= set(names) and set(st["stored_counts"]) <= set(st["stored"])
        labels = [v["man"] for v in vs if v["man"]]
        assert len(set(labels)) == len(labels)
    print("jtest_eval selftest ok")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(s, seeds=False):
        s.add_argument("--split", required=True, choices=("val", "test"))
        s.add_argument("--test-id", default="", help="the run id that the TEST log names (--split test only; the process must run under run.py with that id)")
        s.add_argument("--device", default="cpu")
        s.add_argument("--max", type=int, default=0, help="the first N frames only (a smoke test on VAL; its files go to temp/e343/<split>_max<N>/)")
        if seeds:
            s.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
            s.add_argument("--boot", type=int, default=1000)
        else:
            s.add_argument("--seed", type=int, required=True)
    s = sub.add_parser("dump-b")
    common(s)
    s.add_argument("--b-run", default=None, help="the B-type run to dump (default runs/e273_p_s<seed>; the stack: runs/e292_c2_s<seed>); the output is named predfull_<run>.pkl")
    s.add_argument("--out", default=None)
    s = sub.add_parser("dump-ends")
    common(s)
    s.add_argument("--run", default=None, help="the pair-head run (default e312_p_s<seed>)")
    s.add_argument("--out", default=None)
    s = sub.add_parser("all")
    common(s, seeds=True)
    s.add_argument("--run-prefix", default="e312_p_s", choices=tuple(SETUPS), help="the system: the run is <prefix><seed>")
    s.add_argument("--b-dump", default=None, help="B's dump to reuse ({seed}: the seed); it must exist (default temp/e343/<split>/predfull_e273_p_s<seed>.pkl, reused when it exists, else made)")
    s.add_argument("--control-ends-dump", default=None, help="E312's pair dump, to re-score E312 for its per-frame counts (default: VAL only, the stored dump)")
    s.add_argument("--control-card", default=None, help="E312's stored card (default per split)")
    s.add_argument("--no-stack", action="store_true", help="E324 s without arm k (its check failed)")
    s = sub.add_parser("verify")
    s.add_argument("--seed", type=int, default=None)
    s.add_argument("--run", default=None, help="the run to verify (default e312_p_s<seed>), e.g. e324_s_s0")
    s.add_argument("--boot", type=int, default=0)
    s.add_argument("--no-stack", action="store_true")
    sub.add_parser("selftest")
    a = ap.parse_args()
    if a.cmd == "verify" and (a.run is None) == (a.seed is None):
        ap.error("verify: give --seed or --run")
    for attr in ("out", "b_dump", "control_ends_dump", "control_card"):                            # the script runs in evaluation/ (the launcher's wrapper changes into its folder): paths are relative to the project root
        v = getattr(a, attr, None)
        if v and not os.path.isabs(v):
            setattr(a, attr, os.path.join(ROOT, v))
    if a.cmd == "selftest":
        return selftest()
    if a.cmd == "verify":
        sys.exit(cmd_verify(a))
    if a.cmd == "all":
        sys.exit(cmd_all(a))
    require_gate(a)
    d = dump_dir(a.split, a.max)
    if a.cmd == "dump-b":
        run = os.path.basename(os.path.normpath(a.b_run)) if a.b_run else f"e273_p_s{a.seed}"
        assert re.search(rf"_s{a.seed}$", run), f"--b-run {run}: not a run of seed {a.seed}"
        dump_b(a.split, run, a.out or os.path.join(d, f"predfull_{run}.pkl"), a.device, a.max)
    else:
        run = a.run or f"e312_p_s{a.seed}"
        dump_ends(a.split, run, a.out or os.path.join(d, f"ends_{run}.pkl"), a.device, a.max)


if __name__ == "__main__":
    main()
