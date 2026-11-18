"""read-only CPU exploration for the E264 card: two seeds' raw trees concatenated and assembled by the existing path union (registry assembly settings of fast_recall).
in-sample on VAL 873, a look at the size of the effect only (the card fixes its operating point by a rule on val250)."""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.join(os.environ["BFLC_ROOT"], "evaluation"))
import evaluate as EV
import sparse_metrics as SM
import p16_infer as PI
import p16_train as PT
import p16_builder as PB
R_ = os.environ["BFLC_ROOT"]
reg = EV.registry()
e = reg["fast_recall"]
c = json.load(open(f"{R_}/runs/e123_ladder/cards/{e['run']}.json", encoding="utf-8-sig"))
extra = dict(c["cfg"]); extra.update(PT.parse_set([f"{k}={v}" for k, v in e["set"].items()]))
kw = dict(PT.DESIGNS[e["design"]]); kw.update(extra)
cfg = PB.DefaultCfg(**{k: v for k, v in kw.items() if k not in ("w",)})
A, B = sys.argv[1], sys.argv[2]
ra, gts, _ = EV.builder_raw(reg[A], "val", require_cache=True)
rb, _, _ = EV.builder_raw(reg[B], "val", require_cache=True)

def union(x, y):
    n = len(x["pi"])
    o = {}
    for k in x:
        if k == "parent":
            o[k] = np.concatenate([x[k], np.where(y[k] >= 0, y[k] + n, y[k])])
        elif k == "link_p":
            o[k] = list(x[k]) + [(i + n, j + n, p) for (i, j, p) in y[k]]
        elif k == "junc":
            o[k] = np.maximum(x[k].astype(np.float32), y[k].astype(np.float32)).astype(np.float16)
        else:
            o[k] = np.concatenate([x[k], y[k]])
    return o

U = [union(x, y) for x, y in zip(ra, rb)]
K = ("Q", "lanes_found", "R@1_far", "topo_F1", "F1@1.0", "false_len_frac", "dup_frac", "vert100", "J")
def show(lab, raw, tau, **over):
    asm = dict(e["asm"]); asm.update(over)
    g = [PI.assemble(r, None, tau, cfg, **asm) for r in raw]
    m = SM.card_any(g, gts, extra=True)
    print(f"{lab:28s} tau {tau:.2f} " + " ".join(f"{k} {m[k]:.4f}" for k in K if k in m), flush=True)
show(A, ra, 0.7)
show(B, rb, 0.7)
show(f"{A}+{A} (identity)", [union(x, x) for x in ra[:200]] and [union(x, x) for x in ra], 0.7)
for t in (0.7, 0.8, 0.85):
    show(f"union {A}+{B}", U, t)
