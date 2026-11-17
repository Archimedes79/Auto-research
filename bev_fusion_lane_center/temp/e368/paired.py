"""E368 read-out (scratch): r against z on the VAL stream dumps of one seed, each at its own cell, with the paired scene-bootstrap interval (qgraph_eval.paired + junction) and the other read-out numbers.
usage: python paired.py <seed> <r_run> <r_cell: link,keep,gate(on|off)> <z_run> <z_cell>   e.g. paired.py 0 e368_r_s0 0.3,0.5,on e368_z_s0 0.3,0.5,on"""
import argparse
import os
import sys

ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import jtoken_eval as JE  # noqa: E402

S = int(sys.argv[1])


def load(run, cell):
    lt, kt, g = cell.split(",")
    a = argparse.Namespace(run=run, model="model.pt", seed=S, split="val", every=1, drop=0.0, drop_seed=0, max=0)
    r = JE.read_dump(a, JE.RUNS, 0)
    fr = [JE.decode_variant(f, v, float(lt), float(kt), g == "on", None) for f, v in zip(r["fr"], r["valid"])]
    res = JE.score_preds([JE.frame_prediction(t, f, "rec", float(kt)) for t, f in zip(r["toks"], fr)], r["frames"], log=lambda *x, **k: None)
    return res, r["frames"]


ra, frames = load(sys.argv[2], sys.argv[3])
rb, _ = load(sys.argv[4], sys.argv[5])
print(f"seed {S}: {sys.argv[2]} @ {sys.argv[3]}  vs  {sys.argv[4]} @ {sys.argv[5]}   (VAL {len(frames)} frames, stream dumps)")
for k in JE.KEYS + ("junction_P", "junction_R"):
    print("  %-12s %.4f  %.4f  diff %+.4f" % (k, ra["graph"][k], rb["graph"][k], ra["graph"][k] - rb["graph"][k]))
p = JE.QE.paired(ra["rec"], rb["rec"], frames)
for k in ("topo_F1", "F1@1.0", "lanes_found"):
    if k in p:
        print("  paired %-12s" % k, {x: round(v, 4) for x, v in p[k].items() if x in ("diff", "lo", "hi")})
pj = JE.paired_junction(ra["rec"], rb["rec"], frames)
print("  paired junction", {k: ({x: round(v, 4) for x, v in d.items() if x in ("diff", "lo", "hi")} if isinstance(d, dict) else d) for k, d in pj.items()})
