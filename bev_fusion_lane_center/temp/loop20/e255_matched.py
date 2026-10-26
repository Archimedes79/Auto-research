"""E255 matched-vertex procedure (fixed in the card before the run, equal for every model):

    wsl bash tools/env/_py14.sh temp/loop20/e255_matched.py <registry name> [<registry name> ...]

Layer 1 first: assembling the cached raw outputs at the registry settings must reproduce runs/eval/<name>/val.json (Q within 5e-4, vert100 within 0.01).
Then, on the 250-frame VAL subset (val250, tuning data, declared) sweep tau x vtx_r; per tau interpolate vtx_r linearly to vert100 = 14.3; take the tau with the
highest interpolated Q (upper envelope, no maximum over grid points); evaluate that (tau, interpolated vtx_r) on VAL 873 and write the evaluate.py card as
runs/eval/<name>/val_m143.json (tag m143). Printed: the grid, the chosen point, Q / vert100 / lanes found / R@1 far / topo F1 on VAL 873, on the 623 held-out
frames (873 minus val250) and on val250. Cached raw outputs only (the registry entry must have been evaluated once), 2 processes.
"""
import json
import multiprocessing as mp
import os
import sys

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")
import evaluate as EV  # noqa: E402
import p16_builder as PB  # noqa: E402
import p16_infer as PI  # noqa: E402
import p16_train as PT  # noqa: E402
import sparse_metrics as SM  # noqa: E402

TAUS = (0.12, 0.2, 0.3, 0.5, 0.7)
VTX = (1.5, 2.0, 2.5, 3.0, 3.5, 4.0)
TARGET = 14.3
KEYS = ("Q", "vert100", "lanes_found", "R@1_far", "topo_F1", "F1@1.0")
reg = EV.registry()
KS = np.asarray(EV.subset_ks("val", "val250"))
S = {}  # raw, gts, cfg, asm of the model in work (inherited by the forked workers)


def metrics(idx, tau, vtx):
    asm = dict(S["asm"])
    asm["vtx_r"] = vtx
    g = [PI.assemble(S["raw"][i], None, tau, S["cfg"], **asm) for i in idx]
    m = SM.card_any(g, [S["gts"][i] for i in idx], extra=True)
    return {k: float(m[k]) for k in KEYS}


def point(tv):
    return tv[0], tv[1], metrics(KS, tv[0], tv[1])


def load(name):
    e = reg[name]
    raw, gts, _ = EV.builder_raw(e, "val", None, False, True)
    c = json.load(open(f"{EV.R}/runs/e123_ladder/cards/{e['run']}.json", encoding="utf-8-sig"))
    extra = dict(c["cfg"])
    extra.update(PT.parse_set([f"{k}={v}" for k, v in e["set"].items()]))
    kw = dict(PT.DESIGNS[e["design"]])
    kw.update(extra)
    S.update(raw=raw, gts=gts, cfg=PB.DefaultCfg(**{k: v for k, v in kw.items() if k not in ("w",)}), asm=dict(e["asm"]), tau=e["tau"])


def interp_vtx(rows):
    """vtx_r at which vert100 = TARGET (linear between grid neighbours; vert100 falls with vtx_r); None when TARGET is outside the grid"""
    rows = sorted(rows)
    for (v0, n0, q0), (v1, n1, q1) in zip(rows, rows[1:]):
        if (n0 - TARGET) * (n1 - TARGET) <= 0 and n0 != n1:
            w = (n0 - TARGET) / (n0 - n1)
            return v0 + w * (v1 - v0), q0 + w * (q1 - q0)
    return None


if __name__ == "__main__":
    for name in sys.argv[1:]:
        load(name)
        allidx = np.arange(len(S["raw"]))
        held = np.setdiff1d(allidx, KS)
        stored = json.load(open(f"{EV.R}/runs/eval/{name}/val.json", encoding="utf-8-sig"))["metrics"]
        m0 = metrics(allidx, S["tau"], S["asm"]["vtx_r"])
        ok = abs(m0["Q"] - stored["Q"]) <= 5e-4 and abs(m0["vert100"] - stored["vert100"]) <= 0.01
        print(f"== {name}: layer 1 (registry settings reproduce the stored card): Q {m0['Q']:.4f} vs {stored['Q']:.4f}, vert100 {m0['vert100']:.2f} vs {stored['vert100']:.2f}: {'PASS' if ok else 'FAIL'}")
        if not ok:
            sys.exit(f"{name}: stored card not reproduced")
        with mp.get_context("fork").Pool(2) as p:
            res = p.map(point, [(t, v) for t in TAUS for v in VTX])
        best = None
        for t in TAUS:
            rows = [(v, m["vert100"], m["Q"]) for tt, v, m in res if tt == t]
            print(f"  tau {t:4.2f}: " + "  ".join(f"vtx_r {v:.1f} -> vert100 {n:5.2f} Q {q:.4f}" for v, n, q in sorted(rows)))
            ip = interp_vtx(rows)
            if ip and (best is None or ip[1] > best[2]):
                best = (t, ip[0], ip[1])
        if best is None:
            sys.exit(f"{name}: vert100 {TARGET} is outside the grid for every tau")
        t, v, qi = best
        print(f"  CHOSEN (val250, interpolated at vert100 {TARGET}): tau {t} vtx_r {v:.3f}, Q {qi:.4f}")
        card = EV.evaluate(name, reg, "val", over={"vtx_r": float(v)}, tau=t, tag="m143", require_cache=True)
        for label, idx in (("VAL 873", allidx), ("held-out 623", held), ("val250 (tuning)", KS)):
            m = metrics(idx, t, v)
            print(f"  {label:16} Q {m['Q']:.4f} vert100 {m['vert100']:.2f} lanes_found {m['lanes_found']:.3f} R@1_far {m['R@1_far']:.3f} topo_F1 {m['topo_F1']:.3f} F1@1.0 {m['F1@1.0']:.3f}")
        mm = card["metrics"]
        print(f"  card val_m143.json: Q {mm['Q']:.4f} vert100 {mm['vert100']:.2f} -> {'inside' if abs(mm['vert100'] - TARGET) <= 0.2 else 'OUTSIDE'} 14.3 +- 0.2")
        json.dump(dict(name=name, tau=t, vtx_r=float(v), grid=[dict(tau=a, vtx_r=b, **c) for a, b, c in res]), open(f"{EV.R}/runs/eval/{name}/m143_grid.json", "w"), indent=1)
