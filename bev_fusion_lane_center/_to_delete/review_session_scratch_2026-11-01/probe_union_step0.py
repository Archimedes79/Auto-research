"""throw-away probe: the registry entry e264_u01 (kind union) through evaluate.graphs_of reproduces E264 step 0 (diag_union.py, s0 + s1 at tau 0.7)"""
import evaluate as EV
import sparse_metrics as SM

STEP0 = dict(Q=0.7423, lanes_found=0.7905, **{"R@1_far": 0.7760}, topo_F1=0.7415, false_len_frac=0.0619, dup_frac=0.0902, vert100=16.6495, J=0.4429)
reg = EV.registry()
g, gts, info = EV.graphs_of("ens_j1rc_01", reg["ens_j1rc_01"], "val", require_cache=True)
m = SM.card_any(g, gts, extra=True)
print("cache", info["cache"], "members", list(info["members"]), "asm ms", round(info["asm_ms"], 1))
bad = {k: (round(m[k], 4), v) for k, v in STEP0.items() if abs(m[k] - v) > 5e-4}
print(" ".join(f"{k} {m[k]:.4f}" for k in STEP0), "->", "step 0 reproduced" if not bad else f"DIFFERS {bad}")
