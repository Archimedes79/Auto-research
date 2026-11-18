"""E374 read-out (scratch): the map controls of an arm-m run on VAL, each dump scored at the SAME cell (the one chosen on the run's own TRAIN sample), with the paired scene-bootstrap interval of every mode against the record map (err) and
against the control arm z. usage: python modes.py <seed> <m_run> <z_run> <cell link,keep,gate> [<mode> ...]  e.g. modes.py 0 e374_m_s0 e374_z_s0 0.1,0.6,off clean gross none S U C2"""
import argparse
import os
import sys

ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import jtoken_eval as JE  # noqa: E402

S, m_run, z_run, cell = int(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
modes = sys.argv[5:] or ["clean", "gross", "none", "S", "U", "C2"]


def load(run):
    lt, kt, g = cell.split(",")
    a = argparse.Namespace(run=run, model="model.pt", seed=S, split="val", every=1, drop=0.0, drop_seed=0, max=0)
    r = JE.read_dump(a, JE.RUNS, 0)
    fr = [JE.decode_variant(f, v, float(lt), float(kt), g == "on", None) for f, v in zip(r["fr"], r["valid"])]
    res = JE.score_preds([JE.frame_prediction(t, f, "rec", float(kt)) for t, f in zip(r["toks"], fr)], r["frames"], log=lambda *x, **k: None)
    return res, r["frames"]


rec, frames = load(m_run)
ctl, _ = load(z_run)
print(f"seed {S}, cell {cell}, VAL {len(frames)} frames: topo_F1 / lanes_found / junction_F1 / dup_frac")
fmt = lambda r: "%.4f %.4f %.4f %.4f" % tuple(r["graph"][k] for k in ("topo_F1", "lanes_found", "junction_F1", "dup_frac"))  # noqa: E731
print("  %-8s %s" % ("z", fmt(ctl)))
print("  %-8s %s" % ("err", fmt(rec)))
for m in modes:
    r, _ = load(f"{m_run}_{m}")
    p = JE.QE.paired(r["rec"], rec["rec"], frames)
    pz = JE.QE.paired(r["rec"], ctl["rec"], frames)
    d = lambda q: {x: round(v, 4) for x, v in q["topo_F1"].items() if x in ("diff", "lo", "hi")}  # noqa: E731
    print("  %-8s %s | vs err topo_F1 %s | vs z %s" % (m, fmt(r), d(p), d(pz)))
