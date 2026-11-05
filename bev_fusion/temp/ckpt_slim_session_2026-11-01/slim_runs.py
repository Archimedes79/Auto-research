"""one-off (2026-11-01): slim the checkpoints of FINISHED bev_fusion runs. Run from bev_fusion/.

  slim_runs.py --list                    what would be done (no change)
  slim_runs.py --make-test RUN OUTDIR    a slim copy of the run's final checkpoint in OUTDIR (the run folder is not touched)
  slim_runs.py --apply [RUN ...]         per finished run: slim the final checkpoint, swap it in, check that the run's PINNED code still loads
                                         it (check_load.py), and only then delete the full copy and the older checkpoints; on a failure the
                                         full copy is put back

Slim form = what an old pinned train.py can still load: the optimiser state (Adam moments, 2 x the parameters) is emptied, param_groups and
RNG stay, `slim` = True. Weights, EMA, step and cfg are untouched (compared tensor by tensor before the swap)."""
import glob
import json
import os
import re
import subprocess
import sys

import torch

torch.set_num_threads(2)
HERE = os.path.dirname(os.path.abspath(__file__))
D_CKPTS = "/mnt/d/Auto-research/bev_fusion_ckpts"


def eq(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.shape == b.shape and a.dtype == b.dtype and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(eq(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) == type(b) and len(a) == len(b) and all(eq(x, y) for x, y in zip(a, b))
    try:
        import numpy as np
        if isinstance(a, np.ndarray):
            return isinstance(b, np.ndarray) and np.array_equal(a, b)
    except ImportError:
        pass
    return a == b


def slim_ck(ck):
    out = dict(ck)
    if isinstance(ck.get("opt"), dict) and ck["opt"].get("state"):
        out["opt"] = {"state": {}, "param_groups": ck["opt"]["param_groups"]}
    out["slim"] = True
    return out


def locate(run):
    """-> (folder with the checkpoints, [ckpt files]) : runs/<name> itself, else the D: copy"""
    for d in (run, os.path.join(D_CKPTS, os.path.basename(run))):
        files = sorted(glob.glob(os.path.join(d, "ckpt_*.pt")))
        if files:
            return d, files
    return None, []


def eligible(run):
    d, files = locate(run)
    if not files:
        return None, "no checkpoint"
    cfgp = os.path.join(run, "config.json")
    if not os.path.exists(os.path.join(run, ".done")):
        return None, "no .done flag (unfinished, stopped or old): keep the resume checkpoints"
    if not os.path.exists(cfgp):
        return None, "no config.json"
    step = int(re.search(r"ckpt_(\d+)", files[-1]).group(1))
    if step != json.load(open(cfgp)).get("steps"):
        return None, f"last checkpoint {step} is not the configured end"
    return (d, files), "ok"


def process(run):
    got, why = eligible(run)
    if not got:
        return f"skip: {why}"
    d, files = got
    final = files[-1]
    before = sum(os.path.getsize(f) for f in files)
    ck = torch.load(final, map_location="cpu", weights_only=False)
    sl = slim_ck(ck)
    tmp = final + ".slim.tmp"
    torch.save(sl, tmp)
    back = torch.load(tmp, map_location="cpu", weights_only=False)
    keep = lambda c: {k: v for k, v in c.items() if k not in ("opt", "slim")}          # noqa: E731
    if not eq(keep(ck), keep(back)) or not eq(ck.get("opt", {}).get("param_groups"), back.get("opt", {}).get("param_groups")):
        os.remove(tmp)
        return "FAILED: the slim copy differs from the original in weights, EMA, step, cfg or RNG; nothing changed"
    del ck, back
    os.rename(final, final + ".full")
    os.rename(tmp, final)
    args = [sys.executable, os.path.join(HERE, "check_load.py"), run] + ([d] if d != run else [])
    r = subprocess.run(args, capture_output=True, text=True, env=dict(os.environ, OMP_NUM_THREADS="2", CUDA_VISIBLE_DEVICES=""))
    if r.returncode != 0 or "LOADED" not in r.stdout:
        os.remove(final)
        os.rename(final + ".full", final)
        return "FAILED: the pinned code cannot load the slim checkpoint, the full one is back: " + (r.stderr.strip().splitlines() or ["?"])[-1][:200]
    os.remove(final + ".full")
    for f in files[:-1]:
        os.remove(f)
    for junk in glob.glob(os.path.join(d, "ckpt_*.pt.tmp")):
        os.remove(junk)
    after = os.path.getsize(final)
    return f"slimmed {before / 1e6:.0f} MB -> {after / 1e6:.0f} MB ({len(files)} ckpt -> 1), pinned load ok: {r.stdout.strip().splitlines()[-1]}"


def main():
    a = sys.argv[1:]
    if a[:1] == ["--make-test"]:
        run, out = a[1], a[2]
        d, files = locate(run)
        os.makedirs(out, exist_ok=True)
        torch.save(slim_ck(torch.load(files[-1], map_location="cpu", weights_only=False)), os.path.join(out, os.path.basename(files[-1])))
        print("written", out)
        return
    runs = [r for r in a if not r.startswith("--")] or sorted(p for p in glob.glob("runs/*") if os.path.isdir(p))
    for run in runs:
        got, why = eligible(run)
        if got is None and why == "no checkpoint":
            continue
        if a[:1] == ["--list"]:
            print(f"{os.path.basename(run):26s} {'would slim: ' + str(len(got[1])) + ' ckpt in ' + got[0] if got else 'keep: ' + why}")
        else:
            print(f"{os.path.basename(run):26s} {process(run)}", flush=True)


if __name__ == "__main__":
    main()
