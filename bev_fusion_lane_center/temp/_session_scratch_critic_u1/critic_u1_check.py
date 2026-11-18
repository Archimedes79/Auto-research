"""critic scratch (U1 review): old vs edited jtoken_eval on the same inputs, CPU only. Writes only under temp/_session_scratch_critic_u1/.
1. plain_b(seed 0) of the EDITED module (pred_of_dump without sec0) against the stored runs/e365_planb/planb_s0.pkl and the E324 s base card (check-layer1 without its write)
2. sweep --max N: the old module with --grid perclass, the edited one without --grid, on one CPU dump of the champion head -> the two tables compared
3. score --max N (both decodes): old vs edited, metrics and info (the removed keys listed)"""
import copy
import importlib.util
import json
import os
import pickle
import shutil
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SIMP = os.path.join(ROOT, "temp", "simplify")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 30
WHAT = sys.argv[2].split(",") if len(sys.argv) > 2 else ["planb", "sweep", "score"]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


NEW = load("jtoken_eval", os.path.join(SIMP, "u1_jtoken_eval.py"))
OLD = load("jtoken_eval_old", os.path.join(SIMP, "u1_jtoken_eval.orig.py"))
import numpy as np  # noqa: E402
import torch  # noqa: E402
torch.set_num_threads(2)


def deq(a, b, path="", out=None):
    """exact deep equality -> list of differing paths"""
    out = [] if out is None else out
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b), key=str):
            if k not in a or k not in b:
                out.append(f"{path}/{k} only in {'b' if k not in a else 'a'}")
            else:
                deq(a[k], b[k], f"{path}/{k}", out)
    elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            out.append(f"{path} len {len(a)} vs {len(b)}")
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                deq(x, y, f"{path}[{i}]", out)
    elif isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        if not (isinstance(a, np.ndarray) and isinstance(b, np.ndarray) and a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b, equal_nan=True)):
            out.append(f"{path} array differs")
    elif isinstance(a, float) and isinstance(b, float) and a != a and b != b:
        pass
    elif a != b or type(a) is not type(b):
        out.append(f"{path}: {str(a)[:80]} vs {str(b)[:80]}")
    return out


work = os.path.join(HERE, f"work_n{N}")
if "planb" in WHAT:
    res = NEW.plain_b(0)
    stored = pickle.load(open(os.path.join(ROOT, "runs", "e365_planb", "planb_s0.pkl"), "rb"))
    d = deq(res, stored)
    print("planb: edited plain_b(0) vs runs/e365_planb/planb_s0.pkl:", "IDENTICAL" if not d else f"{len(d)} differences {d[:8]}", flush=True)
    ref = json.load(open(NEW.BASE_CARD.format(seed=0)))["variants"]["primary"]["numbers"]["base"]
    print("planb: compare_numbers vs the E324 s base card:", NEW.compare_numbers(res["graph"], ref), flush=True)
if "sweep" in WHAT or "score" in WHAT:
    runs = os.path.join(work, "runs")
    rd = os.path.join(runs, "e373_cb_s0")
    if not os.path.exists(os.path.join(rd, f"token_val_n{N}.pkl")):
        os.makedirs(rd, exist_ok=True)
        for f in ("model.pt", "config.json"):
            shutil.copy2(os.path.join(ROOT, "runs", "e373_cb_s0", f), rd)
        NEW.dump(NEW.parse_args(["dump", "--run", "e373_cb_s0", "--seed", "0", "--cpu", "--max", str(N)]), runs)
if "sweep" in WHAT:
    o_old, o_new = os.path.join(work, "sweep_old"), os.path.join(work, "sweep_new")
    os.makedirs(o_old)
    os.makedirs(o_new)
    p_old = OLD.sweep(OLD.parse_args(["sweep", "--run", "e373_cb_s0", "--seed", "0", "--split", "val", "--max", str(N), "--grid", "perclass"]), runs, o_old)
    p_new = NEW.sweep(NEW.parse_args(["sweep", "--run", "e373_cb_s0", "--seed", "0", "--split", "val", "--max", str(N)]), runs, o_new)
    a, b = json.load(open(p_old)), json.load(open(p_new))
    d = deq(a, b)
    print("sweep: names", os.path.basename(p_old), os.path.basename(p_new), "| tables", "IDENTICAL" if not d else f"{len(d)} differences {d[:8]}",
          "| row keys equal in order:", all(list(x) == list(y) for x, y in zip(a["rows"], b["rows"])), "| top keys in order:", list(a) == list(b), flush=True)
if "score" in WHAT:
    c_old, c_new = os.path.join(work, "score_old"), os.path.join(work, "score_new")
    os.makedirs(c_old)
    os.makedirs(c_new)
    old = OLD.score(OLD.parse_args(["score", "--run", "e373_cb_s0", "--seed", "0", "--max", str(N)]), runs, c_old)
    new = NEW.score(NEW.parse_args(["score", "--run", "e373_cb_s0", "--seed", "0", "--max", str(N)]), runs, c_new)
    for dec in ("rec", "argmax"):
        a, b = copy.deepcopy(old[dec]), copy.deepcopy(new[dec])
        gone = sorted(set(a["info"]) - set(b["info"]))
        for k in gone:
            a["info"].pop(k)
        for x in (a, b):
            for k in ("date", "commit"):
                x.pop(k, None)
        d = deq(a, b)
        print(f"score {dec}: metrics identical: {a['metrics'] == b['metrics']}; info keys only in the old card: {gone} (old values: {[old[dec]['info'][k] for k in gone]}); the rest:",
              "IDENTICAL" if not d else f"{len(d)} differences {d[:8]}", flush=True)
print("critic_u1_check done", flush=True)
