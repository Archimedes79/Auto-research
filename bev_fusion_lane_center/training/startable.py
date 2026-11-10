"""Is this config startable? Static checks (no GPU, no data read, no training step), the smoke twin and the start line (the BEV system's `training/startable.py`, adapted).

    python training/startable.py <config.json> [--require-clean]
    python training/startable.py <config.json> --write-smoke DIR     also write DIR/smoke_<name>.json (the config with the preset `smoke`: 40 steps, 24 evaluation frames)
    python training/startable.py                                     self-test

A config is the JSON of one run of a design in `DESIGNS` (a table of the keys the trainer of that design reads, with type and range): an unknown key is an error, a wrong type or
a value out of range is an error; the cached inputs the trainer reads must exist (the frozen foundation features `~/exp_gcar_b123/feats`, the split file, the run folder's earlier
state: a finished run is refused, a run with a stored config that differs from this one cannot be resumed), the disk has room, and the uncommitted files of `training/` and
`evaluation/` are listed (the launcher's snapshot is a copy of the working tree, so the arms of a matched control can differ in code); `--require-clean` makes them a problem.
Findings are problems (exit 1) or notes (exit 0). The commands `python -m Training check job.json` and `run job.json` (project.json "training") run this check first.
Run from the problem folder: `wsl bash tools/env/_py14.sh startable.py <config.json>`.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("BFLC_ROOT") or os.path.dirname(HERE)
HOME = os.path.expanduser("~")

# design -> {key: (type(s), check, what)}. `check` is None or a function(value) -> error text / None.
_pos = lambda v: None if v > 0 else "must be > 0"  # noqa: E731
_unit = lambda v: None if 0 < v < 1 else "must be in (0, 1)"  # noqa: E731
DESIGNS = {
    "grid_head": {
        "name": (str, lambda v: None if v and "/" not in v and "." not in v else "a folder name without . and /", "run folder runs/<name> (the id of the start)"),
        "design": (str, None, "this table"),
        "seed": (int, lambda v: None if v >= 0 else "must be >= 0", "seed of the run (weights, batches)"),
        "steps": (int, _pos, "optimiser steps"),
        "bs": (int, _pos, "batch size"),
        "lr": (float, lambda v: None if 0 < v <= 3e-3 else "must be in (0, 3e-3] (training can derail above 6e-4 for the builder; the grid head ran at 1e-3)", "peak learning rate"),
        "width": (int, lambda v: None if v >= 8 and v % 8 == 0 else "must be a multiple of 8 (GroupNorm with 8 groups)", "channel width of the U-Net"),
        "workers": (int, lambda v: None if v >= 0 else "must be >= 0", "data loader workers (2 - 3 while another chat trains)"),
        "budget": (list, lambda v: None if v and all(isinstance(x, (int, float)) and x > 0 for x in v) else "a non-empty list of vertices per 100 m > 0", "read-out budgets, the first is the card's"),
        "per100": ((int, float), _pos, "the vertex budget the loss asks for"),
        "smoke": (bool, None, "40 steps, 24 evaluation frames"),
        "_about": (str, None, "free text"),
    },
    # the lane head of E273 and later (training/train/qgraph_train.py): one command runs the phases in order; a key is the flag of the same name (`node_w_evt` = `--node-w-evt`)
    "qgraph_head": {
        "name": (str, lambda v: None if v and "/" not in v and "." not in v else "a folder name without . and /", "run folder runs/<name> (the phases write runs/<name>_p<k>)"),
        "design": (str, None, "this table"),
        "seed": (int, lambda v: None if v >= 0 else "must be >= 0", "seed of the run; also picks the E271 head runs/e271_a_s<seed> the pixel stage starts from"),
        "phases": (list, lambda v: None if v and all(isinstance(x, int) and 2 <= x <= 7 for x in v) else "a non-empty list of phases 2..7", "phases to run in order (default 2 3 4 5 6 7)"),
        "steps": (list, lambda v: None if v and all(isinstance(x, int) and x > 0 for x in v) else "a list of steps > 0, one per phase", "steps per phase, in the order of phases (default: the trainer's)"),
        "bs": (int, _pos, "batch size"),
        "workers": (int, lambda v: None if v >= 0 else "must be >= 0", "data loader workers (2 - 3 while another chat trains)"),
        "width": (int, lambda v: None if v >= 8 and v % 8 == 0 else "must be a multiple of 8 (GroupNorm with 8 groups)", "channel width"),
        "head": (str, None, "the E271 head to start from (default runs/e271_a_s<seed>/model.pt)"),
        "resume_from": (str, None, "a model.pt of an earlier phase to start from (list only the later phases)"),
        "smoke": (bool, None, "a few steps per phase"),
        "no_gates": (bool, None, "do not stop at a failed gate of a phase (a run by hand, name it in the ledger note)"),
        "gate_phases": (list, None, "only a failed gate of these phases stops the run"),
        "ev_w": ((int, float), _pos, "E274: weight of the 1 m event heat-map loss"),
        "node_w_evt": ((int, float), lambda v: None if v >= 0 else "must be >= 0", "E274: extra weight of the existence positives at event cells"),
        "node_w_inter": ((int, float), lambda v: None if v >= 0 else "must be >= 0", "E274: extra weight of the existence positives inside the intersection mask"),
        "long_match": ((int, float), lambda v: None if v >= 0 else "must be >= 0", "E274: weight of the dual-softmax loss over the long-link candidates"),
        "tier_w": (list, lambda v: None if len(v) == 3 else "three factors", "E276: factor of the positive node terms by GT tier 1 / 2 / 3 (needs the lane GT l4)"),
        "vis_w": ((int, float), lambda v: None if v >= 0 else "must be >= 0", "E276: factor for GT nodes that no camera sees"),
        "neg_w": (list, lambda v: None if len(v) == 4 else "four factors", "E276: factor of the negatives by zone"),
        "link_tier_w": (list, lambda v: None if len(v) == 3 else "three factors", "E276: factor of the link losses by the tier of the source node"),
        "sparse": (list, lambda v: None if len(v) == 2 else "two values: deviation (m), longest chord (m)", "E276: the sparse node target"),
        "node_loss": (str, lambda v: None if v in ("e273", "zone") else "e273 or zone", "E282: the node loss"),
        "zone_bias": ((int, float), lambda v: None if v >= 0 else "must be >= 0", "E282: the tie-break bonus of the zone loss"),
        "soft_nms": (list, lambda v: None if len(v) == 3 else "three values: radius m, smoothness m, passes", "E277: the close-node suppression as a soft step"),
        "_about": (str, None, "free text"),
    },
}
REQUIRED = {"grid_head": ("name", "steps"), "qgraph_head": ("name",)}
FEATS = [f"{HOME}/exp_gcar_b123/feats/{p}_{g}.npy" for p in ("train", "val") for g in ("dec32", "logit7")]
INPUTS = {"grid_head": FEATS + ["data/splits.json"], "qgraph_head": FEATS + ["data/splits.json"]}
SMOKE = dict(steps=40, smoke=True)
SMOKE_OF = {"qgraph_head": dict(smoke=True)}                       # the steps of a qgraph run are a list per phase: the trainer's own smoke values apply


def inputs_of(cfg):
    """the files and folders the trainer of cfg's design reads (relative paths are relative to the problem folder)"""
    design = cfg.get("design")
    out = list(INPUTS.get(design, []))
    if design == "qgraph_head" and not (cfg.get("head") or cfg.get("resume_from")):
        out.append(f"runs/e271_a_s{cfg.get('seed', 0)}/model.pt")      # the pixel stage starts from the E271 head of the same seed
    return out


def lane_store_note(root=ROOT):
    """a note (not a problem: the GT chat owns the store) when the lane GT store of this checkout is missing or empty; None when it is there or cannot be asked"""
    try:
        sys.path.insert(0, os.path.dirname(root))
        from GtGenerator import lane_gt
        sig_dir = os.path.join(lane_gt.DATA, lane_gt.signature())
        n = len(lane_gt.held(sig_dir)) if os.path.isdir(sig_dir) else 0
    except Exception as e:  # noqa: BLE001
        return f"the lane GT store was not checked ({type(e).__name__})"
    return None if n else f"the lane GT store {os.path.basename(sig_dir)} of this GtGenerator is missing or empty (`python -m GtGenerator key lane_center --set full`)"


def check(cfg, require_clean=False, root=ROOT, inputs=None):
    """(problems, notes) for a config dict"""
    problems, notes = [], []
    design = cfg.get("design")
    if design not in DESIGNS:
        return [f"design {design!r} is not one of {sorted(DESIGNS)}"], notes
    table = DESIGNS[design]
    for k, v in cfg.items():
        if k not in table:
            problems.append(f"unknown key {k!r} (a typo? the keys of design {design}: {', '.join(sorted(table))})")
            continue
        typ, fn, _ = table[k]
        ok = isinstance(v, typ) and not (typ is int and isinstance(v, bool))
        if not ok:
            problems.append(f"{k}: {v!r} is not {typ if isinstance(typ, type) else '/'.join(t.__name__ for t in typ)}")
        elif fn is not None and fn(v):
            problems.append(f"{k}: {v!r} {fn(v)}")
    for k in REQUIRED[design]:
        if k not in cfg:
            problems.append(f"missing key {k!r}")
    if inputs is None and design == "qgraph_head":
        n = lane_store_note(root)
        if n:
            notes.append(n)
    for p in (inputs if inputs is not None else inputs_of(cfg)):
        full = p if os.path.isabs(p) else os.path.join(root, p)
        if not os.path.exists(full):
            problems.append(f"input missing: {p}")
    name = cfg.get("name")
    if name:
        rdir = os.path.join(root, "runs", name)
        if os.path.exists(os.path.join(rdir, ".done")):
            problems.append(f"runs/{name} is a finished run (.done): run.py refuses the id unless --again")
        sc = os.path.join(rdir, "config.json")
        if os.path.exists(sc) and not os.path.exists(os.path.join(rdir, ".done")):
            try:
                stored = json.load(open(sc, encoding="utf-8"))
                vals = dict(stored.get("args") or {})                      # the trainer writes the flat config and the command line of the run under "args"
                vals.update({k: v for k, v in stored.items() if k != "args"})
                diff = sorted(k for k in ("steps", "lr", "bs", "width", "seed", "per100", "budget") if k in cfg and k in vals and cfg[k] != vals[k])
                if diff:
                    problems.append(f"runs/{name} has a run with another {', '.join(diff)}: a resume must use the same config (change the id for a new arm)")
                else:
                    notes.append(f"runs/{name} has a run: the same command again resumes it from its last checkpoint")
            except (OSError, ValueError):
                notes.append(f"runs/{name}/config.json is unreadable")
    free = shutil.disk_usage(root).free / 1e9
    if free < 5:
        problems.append(f"only {free:.1f} GB free on the disk of the project")
    try:
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "training", "evaluation"], cwd=root, capture_output=True, text=True, timeout=60).stdout.split("\n")
        dirty = [ln[3:] for ln in dirty if ln.strip()]
    except Exception:  # noqa: BLE001
        dirty = []
    if dirty:
        (problems if require_clean else notes).append(f"{len(dirty)} uncommitted file(s) in training/ evaluation/ go into the run's snapshot: {', '.join(dirty[:6])}{' ...' if len(dirty) > 6 else ''}")
    return problems, notes


def smoke_of(cfg):
    out = dict(cfg)
    out.update(SMOKE_OF.get(cfg.get("design"), SMOKE))
    if cfg.get("design") == "qgraph_head":
        out.pop("steps", None)
    out["name"] = "smoke_" + cfg["name"]
    return out


def resolve(path):
    """a config path as given, else relative to the problem folder (`_py14.sh` runs a script in its own folder, `Training check` / `run` name the config relative to the problem folder)"""
    return path if os.path.isfile(path) else os.path.join(ROOT, path)


def selftest():
    import tempfile
    tmp0 = tempfile.mkdtemp()                                   # every check below runs on an empty problem folder: a real run of the same name must not change the result
    good = dict(name="e271_a_s0", design="grid_head", seed=0, steps=4000, bs=8, lr=1e-3, width=48, workers=3, budget=[11.2, 8.0, 14.3], per100=11.2)
    p, _ = check(good, root=tmp0, inputs=[])
    assert not p, p
    for bad, frag in (({"stepz": 1}, "unknown key"), ({"steps": 0}, "must be > 0"), ({"steps": "9"}, "not"), ({"width": 50}, "multiple of 8"), ({"lr": 0.1}, "(0, 3e-3]"),
                      ({"budget": []}, "non-empty"), ({"design": "x"}, "not one of"), ({"name": "a.b"}, "folder name"), ({"seed": True}, "not")):
        p, _ = check(dict(good, **bad), root=tmp0, inputs=[])
        assert any(frag in x for x in p), (bad, p)
    p, _ = check({k: v for k, v in good.items() if k != "steps"}, root=tmp0, inputs=[])
    assert any("missing key" in x for x in p)
    p, _ = check(good, root=tmp0, inputs=["no/such/input.npy"])
    assert any("input missing" in x for x in p)
    q = dict(name="e273_p_s0", design="qgraph_head", seed=0, workers=3, node_w_evt=3.0, sparse=[0.3, 7.5])
    assert not check(q, root=tmp0, inputs=[])[0]
    for bad, frag in (({"phases": [1]}, "phases 2..7"), ({"tier_w": [1, 2]}, "three factors"), ({"node_loss": "x"}, "e273 or zone"), ({"steps": [0]}, "steps > 0"), ({"stepz": 1}, "unknown key")):
        p, _ = check(dict(q, **bad), root=tmp0, inputs=[])
        assert any(frag in x for x in p), (bad, p)
    assert "runs/e271_a_s1/model.pt" in inputs_of(dict(q, seed=1)) and "runs/e271_a_s0/model.pt" not in inputs_of(dict(q, head="x.pt"))
    qs = smoke_of(q)
    assert qs["smoke"] and "steps" not in qs and qs["name"] == "smoke_e273_p_s0" and not check(qs, root=tmp0, inputs=[])[0]
    s = smoke_of(good)
    assert s["steps"] == 40 and s["smoke"] and s["name"] == "smoke_e271_a_s0" and not check(s, root=tmp0, inputs=[])[0]
    tmp = tempfile.mkdtemp()
    os.makedirs(os.path.join(tmp, "runs", "e271_a_s0"))
    open(os.path.join(tmp, "runs", "e271_a_s0", ".done"), "w").close()
    p, _ = check(good, root=tmp, inputs=[])
    assert any("finished run" in x for x in p), p
    os.remove(os.path.join(tmp, "runs", "e271_a_s0", ".done"))
    json.dump(dict(good, steps=3000), open(os.path.join(tmp, "runs", "e271_a_s0", "config.json"), "w"))
    p, _ = check(good, root=tmp, inputs=[])
    assert any("another steps" in x for x in p), p
    json.dump(good, open(os.path.join(tmp, "runs", "e271_a_s0", "config.json"), "w"))
    p, n = check(good, root=tmp, inputs=[])
    assert not any("another" in x for x in p) and any("resumes" in x for x in n), (p, n)
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.rmtree(tmp0, ignore_errors=True)
    print("startable selftest passed")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("config", nargs="?")
    ap.add_argument("--require-clean", action="store_true")
    ap.add_argument("--write-smoke", metavar="DIR")
    a = ap.parse_args()
    if not a.config:
        return selftest()
    cfg = json.load(open(resolve(a.config), encoding="utf-8"))
    problems, notes = check(cfg, a.require_clean)
    for n in notes:
        print("note:", n)
    for p in problems:
        print("PROBLEM:", p)
    if a.write_smoke and not problems:
        os.makedirs(a.write_smoke, exist_ok=True)
        s = smoke_of(cfg)
        path = os.path.join(a.write_smoke, f"smoke_{cfg['name']}.json")
        json.dump(s, open(path, "w", encoding="utf-8"), indent=1)
        print("smoke config:", path)
    print("startable" if not problems else f"NOT startable: {len(problems)} problem(s)")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
