"""critic scratch (U1 review): plain_b(seed) of the OLD and the EDITED module (deep equality: U1's claim), and both against the stored runs/e365_planb/planb_s<S>.pkl
(graph_numbers of every key, junction numbers, and the paired interval new - stored: does a re-run of check-layer1 change what `score` reads?). CPU only, writes nothing."""
import importlib.util
import os
import pickle
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SIMP = os.path.join(ROOT, "temp", "simplify")
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


NEW = load("jtoken_eval", os.path.join(SIMP, "u1_jtoken_eval.py"))
OLD = load("jtoken_eval_old", os.path.join(SIMP, "u1_jtoken_eval.orig.py"))
import torch  # noqa: E402
torch.set_num_threads(2)
import numpy as np  # noqa: E402


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


quiet = lambda *a, **k: None  # noqa: E731
rn = NEW.plain_b(SEED)
ro = OLD.plain_b(SEED)
d = deq(ro, rn)
print(f"seed {SEED}: OLD plain_b vs EDITED plain_b:", "IDENTICAL" if not d else f"{len(d)} differences {d[:8]}", flush=True)
st = pickle.load(open(os.path.join(ROOT, "runs", "e365_planb", f"planb_s{SEED}.pkl"), "rb"))
QE = NEW.QE
gs, gn = QE.graph_numbers(st["rec"]["graph"]), QE.graph_numbers(rn["rec"]["graph"])
dk = {k: gn[k] - gs[k] for k in gs if k in gn and isinstance(gs[k], float) and gs[k] == gs[k] and gn[k] != gs[k]}
print("graph_numbers stored vs re-made: keys", len(gs), "differing", dk, "| only stored:", sorted(set(gs) - set(gn))[:10], "| only re-made:", sorted(set(gn) - set(gs))[:10], flush=True)
print("graph (the card numbers) stored vs re-made:", deq(st["graph"], rn["graph"])[:8], flush=True)
frames = QE.val_frames()
p = QE.paired(rn["rec"], st["rec"], frames)
nz = {k: (v["diff"], v["lo"], v["hi"]) for k, v in p.items() if isinstance(v, dict) and "diff" in v and (v["diff"] != 0 or v["lo"] != 0 or v["hi"] != 0)}
print("paired re-made - stored: keys", len(p), "non-zero", nz, flush=True)
print("junction stored vs re-made:", NEW.junction_numbers(st["rec"]["graph"]), NEW.junction_numbers(rn["rec"]["graph"]), flush=True)
print("critic_u1_planb done", flush=True)
