"""Read-only scratch (past-critic agent, 2026-10-31): tau frontier of the recall-weight family on cached VAL raw outputs + per-piece DP vertex count.
Writes only to stdout. Uses evaluate.py's assembly path (same cfg construction) without touching the registry or runs/eval."""
import os, sys, json, pickle, time
R = os.environ["BFLC_ROOT"]
sys.path.insert(0, f"{R}/evaluation")
import evaluate as EV
import sparse_metrics as SM
import p16_infer as PI
import p16_train as PT
import p16_builder as PB

HOME = os.path.expanduser("~")
P18 = f"{HOME}/exp_gcar_b123/p18"
ASM = {"cov_r": 0.5, "vtx_r": 1.5, "junc_thr": 0.5, "reloc_thr": 0.4, "reloc_shift": 12, "reloc_tol": 1.6, "fork_prune": 0.4, "fast": 2}
K = ("Q", "lanes_found", "R@1_far", "P@1_far", "topo_F1", "F1@1.0", "false_len_frac", "dup_frac", "vert100")


def cfg_of(card_run, M, K0):
    c = json.load(open(f"{R}/runs/e123_ladder/cards/{card_run}.json", encoding="utf-8-sig"))
    extra = dict(c["cfg"])
    extra.update(PT.parse_set([f"M={M}", f"K0={K0}"]))
    kw = dict(PT.DESIGNS["D7"])
    kw.update(extra)
    return PB.DefaultCfg(**{k: v for k, v in kw.items() if k not in ("w",)})


def line(tag, g, gts):
    m = SM.card_any(g, gts, extra=True)
    dp = SM.vert100_dp(g, 0.3)
    print(tag, " ".join(f"{k} {m[k]:.4f}" for k in K if k in m), f"vdp {dp:.2f}", flush=True)


models = sys.argv[1].split(",")
taus = [float(t) for t in sys.argv[2].split(",")]
over = dict(a.split("=") for a in sys.argv[3:])
asm = dict(ASM)
asm.update({k: float(v) for k, v in over.items()})
for spec in models:
    if spec == "control":
        reg = EV.registry()
        g, gts, _ = EV.graphs_of("control", reg["control"], "val")
        line("control", g, gts)
        continue
    run, M, cardrun = spec.split(":")
    M = int(M)
    raw, gts = pickle.load(open(f"{P18}/raw_val873_{run}_{M}.pkl", "rb"))
    cfg = cfg_of(cardrun, M, M // 2)
    for t in taus:
        t0 = time.time()
        g = [PI.assemble(r, None, t, cfg, **asm) for r in raw]
        line(f"{run} tau {t} {over}", g, gts)
    del raw
