"""Checks of the lane suite on the real VAL data (evaluation/lane_suite.py): the adapter is aligned with the project's data, the suite's numbers
equal the project's own, and every function runs end to end.   tools/env/_py14.sh lane_suite.py selftest [--quick]

  1 orientation   the stored dense maps are [row = x, col = y]; with the adapter's transposition the maps lie on the ground-truth lines (tolerant F1 >> the other
                  orientations), so a mix-up of axes cannot go unnoticed
  2 alignment     the ground truth of the graph subject is the ground truth of the source, row by row
  3 the project's numbers   suite lcr.* / lcr_topo.* of graphs:champion on ALL VAL frames = the stored card runs/eval/champion/val.json (tolerance 5e-4, like the
                  golden check); --quick: on the first scenes against evaluate.py's own functions
  4 common metrics   suite geo.* (P / R / F1 at 0.5 / 1.0 / 1.5 m, lanes found) = the project's geometry numbers on the same frames
  5 contract        the adapter contract checker on both systems
  6 end to end      dense head: quant run -> table -> outliers -> probe scan -> a drawn frame, on 3 scenes
"""
import bflc_paths as BP  # project paths from the location of this checkout
import json
import os
import sys
import tempfile
import time
import traceback

import numpy as np

import lane_suite as LS
from evalsuite import EvalSuite
from lane_task import measures as M

CARD = f"{BP.ROOT}/runs/eval/champion/val.json"
KEYS_PROJECT = ("S", "Q", "F1@0.5", "F1@1.0", "F1@1.5", "J", "lanes_found", "vert100", "false_len_frac", "len_ratio")
KEYS_TOPO = ("topo_F1", "topo_P", "topo_R", "dup_frac", "R@1_far", "R@1_mid", "R@1_near")


def t_orientation(quick):
    src, subj, task = LS.build("dense:e37_final")
    cfg = task.cfg
    kw = dict(grid=cfg.grid, thr=cfg.thr, taus=cfg.taus, sweep=cfg.sweep)
    res = {}
    for name, tf in (("adapter (transposed)", lambda a: a), ("as stored", lambda a: a.T), ("flipped", lambda a: a[::-1].copy())):
        cs = []
        for sc in src.scenes()[:6]:
            for ref in src.frames(sc)[::5]:
                s = src.load(ref)
                cs.append(M.image_counts(tf(subj.predict(s).outputs["prob"]), s.reference["mask"], **kw))
        res[name] = M.image_numbers({k: sum(c[k] for c in cs) for k in cs[0]}, cfg.taus, cfg.sweep)["F1_rel@1.0"]
    print(f"      tolerant F1 at 1 m: " + ", ".join(f"{k} {v:.3f}" for k, v in res.items()))
    assert res["adapter (transposed)"] > 0.85 and res["adapter (transposed)"] - max(res["as stored"], res["flipped"]) > 0.3, res


def t_alignment(quick):
    src, subj, task = LS.build("graphs:champion")
    bad = 0
    for sc in src.scenes()[: 3 if quick else None]:
        for ref in src.frames(sc):
            s = src.load(ref)
            g = subj.gts[s.inputs["row"]]
            a, b = s.reference["raw"], g
            same = len(a["pts"]) == len(b["pts"]) and all(np.allclose(x, y) for x, y in zip(a["pts"], b["pts"]))
            bad += not same
    assert bad == 0, f"{bad} frames: the ground truth of the source differs from the one the model was evaluated on"
    assert len(src.rows) == 873 and len(src.scenes()) == 22


def _project_direct(src, subj, labels):
    import bootstrap as SB
    import topo_metrics as TM
    preds = [subj.graphs[src.load(r).inputs["row"]] for r in labels]
    gts = [src.load(r).reference["raw"] for r in labels]
    return SB.pooled(SB.frame_records(preds, gts)), TM.pooled([TM.frame_record(p, g) for p, g in zip(preds, gts)])


def t_project_numbers(quick):
    src, subj, task = LS.build("graphs:champion")
    with tempfile.TemporaryDirectory() as d:
        s = EvalSuite(src, subj, task, out=d, state="reset")
        scenes = src.scenes()[:3] if quick else None
        t0 = time.time()
        st = s.quant.run(scenes=scenes, name="proj", progress=200)
        tab = s.quant.table(st)
        print(f"      {len(st.frames)} frames, {time.time() - t0:.0f} s")
        if quick:
            refs = [src.frames(sc)[i] for sc in scenes for i in range(len(src.frames(sc)))]
            ref_m, ref_t = _project_direct(src, subj, refs)
            what = "evaluate.py's own functions on the same frames"
        else:
            card = json.load(open(CARD))["metrics"]
            ref_m, ref_t = card, card
            what = "the stored card runs/eval/champion/val.json"
        worst = 0.0
        for k in KEYS_PROJECT:
            if k in ref_m:
                worst = max(worst, abs(tab[f"lcr.{k}"] - ref_m[k]) / max(1.0, abs(ref_m[k])))
        for k in KEYS_TOPO:
            if k in ref_t and ref_t[k] == ref_t[k]:
                worst = max(worst, abs(tab[f"lcr_topo.{k}"] - ref_t[k]) / max(1.0, abs(ref_t[k])))
        print(f"      lcr.* / lcr_topo.* against {what}: largest relative difference {worst:.2e}")
        assert worst < 5e-4, worst
        # common metrics against the project's geometry numbers on the same frames (the general code, written independently)
        diffs = {}
        for t in ("0.5", "1.0", "1.5"):
            for p in ("P", "R", "F1"):
                diffs[f"{p}@{t}"] = abs(tab[f"geo.{p}@{t}"] - tab[f"lcr.{p}@{t}"])
        diffs["lanes_found"] = abs(tab["geo.lanes_found"] - tab["lcr.lanes_found"])
        w = max(diffs.values())
        print(f"      geo.* (general suite code) against lcr.* (project code): largest difference {w:.4f} ({max(diffs, key=diffs.get)})")
        assert w < 0.01, diffs


def t_contract(quick):
    with tempfile.TemporaryDirectory() as d:
        for system in ("dense:e37_final", "graphs:champion"):
            src, subj, task = LS.build(system)
            res = EvalSuite(src, subj, task, out=d).check()
            assert all(r["ok"] for r in res), (system, [r for r in res if not r["ok"]])


def t_end_to_end(quick):
    src, subj, task = LS.build("dense:e37_final")
    with tempfile.TemporaryDirectory() as d:
        s = EvalSuite(src, subj, task, out=d, state="reset")
        st = s.quant.run(scenes=src.scenes()[:3], name="dense", progress=200)
        tab = s.quant.table(st)
        print("      " + "  ".join(f"{k} {tab[k]:.3f}" for k in ("img.F1_rel@1.0", "img.AP_rel", "geo.F1@1.0", "geo.chamfer", "geo.lanes_found", "topo.topo_F1")))
        assert tab["img.F1_rel@1.0"] > 0.85 and tab["geo.F1@1.0"] > 0.7
        assert s.quant.outliers(st, k=3)["frames"]
        res = s.probe.scan(st, k=1, top=2, seed=0)
        assert os.path.exists(res["index"]) and res["budget"]
        print("      where to work:", {k: round(v, 2) for k, v in res["targets"].items()})
        q = s.qual.frame(src.frames(src.scenes()[0])[4].label, ("overview",))
        assert all(os.path.exists(p) for p in q.images.values())


def main(argv) -> int:
    quick = "--quick" in argv
    checks = [("orientation of the dense maps", t_orientation), ("alignment of the ground truth", t_alignment),
              ("the project's numbers reproduced + common metrics agree", t_project_numbers), ("adapter contract", t_contract),
              ("end to end on the dense head", t_end_to_end)]
    bad, t0 = 0, time.time()
    for name, fn in checks:
        t = time.time()
        try:
            fn(quick)
            print(f"ok    {name}  ({time.time() - t:.0f} s)")
        except Exception:  # noqa: BLE001
            bad += 1
            print(f"FAIL  {name}\n{traceback.format_exc()}")
    print(f"lane_suite selftest{' (quick)' if quick else ''}: {len(checks) - bad}/{len(checks)} checks passed in {time.time() - t0:.0f} s")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
