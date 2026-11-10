"""Front-area comparison of the detections of several systems: `bev_suite.py front` (the front-camera sector, x in [0, X_MAX) m and +-32.3 deg).

    python evaluation/bev_suite.py front REF SYSTEM [SYSTEM ...] --n 1000 [--x-max 50] [--tag ""] [--det det] [--boot 1000] [--out FILE.md] [--force]

A system is a run folder (`runs/<name>`, read from its `eval/scene_records*.pkl`, the file `bev_suite.py eval` writes) or `LABEL=FILE.pkl` (the same
record of something that is no run of ours, e.g. a front-camera-only reference network). The first system is the reference. Per system: AP at
0.5 / 1 / 2 / 4 m for vehicle and pedestrian, mAP2 = their mean, mAP2 per range bin, and for every other system the difference to the reference with
the paired scene bootstrap (`Evaluation.stats.scene_picks`, the one random stream of the research system). Why a sector: a camera-only reference sees
only the front camera, so every system is scored on the boxes in that sector (a front-area comparison of 2026-10-31 had been made by a script in
temp/t4, which this replaces). X_MAX may not exceed the last range bin (60 m).

How: `metrics.DetMeter.frame_rows` gives the true-positive flags of every frame once (the greedy matching involves only the frame's own boxes); a
resample re-sorts the rows of the drawn scenes, it does not re-match. Identity checks, all refused with `--force` turned into warnings: the recorded GT
signatures (evaluation and training) and the frame sets, as `bev_suite.py compare` (a `LABEL=FILE.pkl` takes them from the pickle's meta when it has
any); the scenes and the number of frames per scene; and the number of ground-truth boxes in the sector, which must be the same for every system (the
same frames under the same GT give the same count). A record without any recorded identity gets a warning, and only the last two checks apply.
A range bin shows the mean over the classes that have ground truth in it (`*`: only some of them), the headline mAP2 needs both.

Run the self-test: python evaluation/front_compare.py   (CPU, seconds)
"""
from __future__ import annotations

import argparse
import math
import os
import pickle
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
for _p in (ROOT, os.path.join(ROOT, "..", "autoresearch")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from evaluation.metrics import DIST_THRESHOLDS, FRONT_HALF_FOV, RANGE_BINS, DetMeter, bin_label  # noqa: E402

CLASSES = ("vehicle", "pedestrian")                      # mAP2 = the mean over these two classes and the four distances


def load_records(path, det="det", dt=0.0):
    """scene_records pickle -> ({scene: [(dets, gt boxes) per frame]} of the box set `det` ('det' plain, 'detr' refined) at target time dt, meta)"""
    with open(path, "rb") as f:
        rec = pickle.load(f)
    return {s: r[dt] for s, r in rec[det].items() if dt in r}, rec.get("meta") or {}


def scene_rows(frames, meter, classes=CLASSES):
    """{scene: one additive 'frame' for DetMeter.reduce_rows}: the rows and GT counts of all frames of the scene, in frame order (the boxes outside
    the scored sector are dropped by meter.keep)"""
    out = {}
    for s, fr in frames.items():
        rows = [meter.frame_rows(meter.keep(ds), meter.keep(gs), list(classes)) for ds, gs in fr]
        out[s] = {"rows": np.concatenate([r["rows"] for r in rows]) if rows else np.zeros((0, meter.n_cols)),
                  "n_gt": sum((r["n_gt"] for r in rows), np.zeros((len(classes), 1 + len(meter.bins)), int))}
    return out


def map2(res, classes=CLASSES, scope="all", thresholds=DIST_THRESHOLDS, need_all=True):
    """mean AP over the classes and distance thresholds of one reduce_rows result. need_all: a class without ground truth in the scope makes it
    nan (the headline); otherwise such a class is left out (a range bin), nan when no class has any."""
    use = [c for c in classes if need_all or res[c][scope]["n_gt"]]
    v = [res[c][scope]["ap@%g" % t] for c in use for t in thresholds]
    return float(np.mean(v)) if v and all(x == x for x in v) else float("nan")


def compare(systems, x_max=50.0, n_boot=1000, seed=0, classes=CLASSES, strict_gt=True):
    """systems: [(label, {scene: [(dets, gts)]})], the first is the reference; every system holds the same scenes and as many frames per scene, and (strict_gt)
    the same number of ground-truth boxes in the sector. -> {'scenes', 'frames', 'x_max', 'n_boot', 'seed', 'n_gt' (reference), 'warnings',
    'systems': [{'label', 'map2', 'ap' {class: {threshold: AP}}, 'bins' {label: mAP2 over the classes with ground truth}, 'partial' [bin labels with
    only some classes], 'draws' (mAP2 of every resample), 'diff', 'lo', 'hi', 'n_valid'} ...]}; diff / lo / hi: system - reference, None for the reference"""
    from Evaluation.stats import percentile_interval, scene_picks
    if x_max > RANGE_BINS[-1][1]:
        raise ValueError(f"x_max {x_max:g} m is beyond the last range bin ({RANGE_BINS[-1][1]:g} m): the boxes behind it are in no bin")
    labels = [lab for lab, _ in systems]
    if len(set(labels)) != len(labels):
        raise ValueError(f"system labels must differ: {labels}")
    ref_lab, ref = systems[0]
    scenes = sorted(ref)
    if not scenes or not any(ref[s] for s in scenes):
        raise ValueError(f"{ref_lab} holds no frames of the requested box set / target time (an empty record: e.g. --det detr on a run without refined boxes, or an absent --dt)")
    for lab, fr in systems[1:]:
        if sorted(fr) != scenes:
            raise ValueError(f"{lab} and {ref_lab} hold different scenes ({len(set(fr) & set(ref))} common of {len(fr)} / {len(ref)})")
        bad = [s for s in scenes if len(fr[s]) != len(ref[s])]
        if bad:
            raise ValueError(f"{lab} and {ref_lab} hold different frame counts in {len(bad)} scenes (first: {bad[0]}: {len(fr[bad[0]])} / {len(ref[bad[0]])})")
    meter = DetMeter(classes=list(classes), x_max=x_max)
    bins = [i for i, (lo, _) in enumerate(RANGE_BINS) if lo < x_max]
    rows = {lab: scene_rows(fr, meter, classes) for lab, fr in systems}
    n_gt = {lab: sum((rows[lab][s]["n_gt"] for s in scenes), np.zeros((len(classes), 1 + len(meter.bins)), int)) for lab in rows}
    warnings = []
    for lab in labels[1:]:
        if not np.array_equal(n_gt[lab], n_gt[ref_lab]):
            msg = (f"{lab} has {n_gt[lab][:, 0].tolist()} ground-truth boxes in the sector ({', '.join(classes)}), {ref_lab} has {n_gt[ref_lab][:, 0].tolist()}: "
                   "not the same frames under the same GT")
            if strict_gt:
                raise ValueError(msg)
            warnings.append("WARNING (--force): " + msg)
    only_all = {(c, "all") for c in classes}
    picks = list(scene_picks(len(scenes), n_boot, seed))             # the same draws for every system: the difference is paired
    boot = {lab: [map2(meter.reduce_rows([rows[lab][scenes[i]] for i in ix], list(classes), only=only_all)) for ix in picks] for lab in rows}
    out = {"scenes": len(scenes), "frames": sum(len(ref[s]) for s in scenes), "x_max": x_max, "n_boot": n_boot, "seed": seed, "warnings": warnings,
           "n_gt": {c: int(n_gt[ref_lab][ci, 0]) for ci, c in enumerate(classes)}, "systems": []}
    for lab, _ in systems:
        res = meter.reduce_rows([rows[lab][s] for s in scenes], list(classes))
        entry = {"label": lab, "map2": map2(res, classes), "ap": {c: {t: res[c]["all"]["ap@%g" % t] for t in DIST_THRESHOLDS} for c in classes},
                 "bins": {bin_label(i): map2(res, classes, bin_label(i), need_all=False) for i in bins},
                 "partial": [bin_label(i) for i in bins if 0 < sum(1 for c in classes if res[c][bin_label(i)]["n_gt"]) < len(classes)],
                 "draws": boot[lab], "diff": None, "lo": None, "hi": None, "n_valid": None}
        if lab != ref_lab:
            lo, hi, nv = percentile_interval(np.array(boot[lab]) - np.array(boot[ref_lab]))
            entry.update(diff=entry["map2"] - out["systems"][0]["map2"], lo=lo, hi=hi, n_valid=nv)
        out["systems"].append(entry)
    return out


def markdown(res, sources=(), notes=(), classes=CLASSES, records=""):
    """the result of compare() as the text of a results file; sources: lines naming the record of every system; notes: warnings of the checks;
    records: what the records are (split, start mode)"""
    ref = res["systems"][0]["label"]
    fov = math.degrees(FRONT_HALF_FOV)
    L = [f"# Front-area comparison (x in [0, {res['x_max']:g}) m, +-{fov:.1f} deg), {records + ', ' if records else ''}{res['frames']} frames of {res['scenes']} scenes", "",
         "AP by centre distance over " + " / ".join(f"{t:g}" for t in DIST_THRESHOLDS) + " m (nuScenes style), mAP2 = mean of "
         + " and ".join(classes) + f" AP over the four distances; interval of the difference to `{ref}` = paired scene bootstrap "
         f"({res['n_boot']} draws, seed {res['seed']}, 2.5 / 97.5 %).", ""]
    notes = list(notes) + list(res.get("warnings", []))
    L += [f"- {s}" for s in sources] + [f"- {n}" for n in notes] + ([""] if sources or notes else [])
    head = ["system", "mAP2", "d vs ref [95% interval]"] + [f"{c[:3]} AP@{t:g}" for c in classes for t in DIST_THRESHOLDS]
    L += ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for s in res["systems"]:
        if s["diff"] is None:
            d = "-"
        else:
            d = f"{s['diff']:+.3f} [{s['lo']:+.3f}, {s['hi']:+.3f}]" + (f" ({s['n_valid']} of {res['n_boot']} draws scorable)" if s["n_valid"] < res["n_boot"] else "")
        L.append(f"| {s['label']} | {s['map2']:.3f} | {d} | " + " | ".join(f"{s['ap'][c][t]:.3f}" for c in classes for t in DIST_THRESHOLDS) + " |")
    labs = list(res["systems"][0]["bins"])
    shown = [f"{float(b.split('-')[0]):g}-{min(float(b.split('-')[1]), res['x_max']):g} m" for b in labs]      # the last bin ends at the sector
    L += ["", "mAP2 by distance of the object (range bins of the metric; n/a: no ground truth in the bin" + ("; *: only some classes have" if any(s["partial"] for s in res["systems"]) else "") + "):", "",
          "| system | " + " | ".join(shown) + " |", "|---|" + "---|" * len(labs)]
    for s in res["systems"]:
        L.append(f"| {s['label']} | " + " | ".join("n/a" if s["bins"][b] != s["bins"][b] else f"{s['bins'][b]:.3f}" + ("*" if b in s["partial"] else "") for b in labs) + " |")
    L += ["", "GT boxes in the area (" + ref + "): " + ", ".join(f"{c} {n}" for c, n in res["n_gt"].items()) + "."]
    return "\n".join(L) + "\n"


def _ident(arg, tag, n, det, dt):
    """(label, records path, {scene: frames}, info, training GT signature, is a run) of a command line system. info = the tuple of
    bev_suite.legacy_record (file tag, frame keys, number of frames, evaluation GT signature), for LABEL=FILE.pkl the same from the pickle's meta, or None
    when nothing is recorded"""
    from evaluation import bev_suite
    if "=" in arg and not os.path.isdir(arg):
        label, path = arg.split("=", 1)
        if not os.path.isfile(path):
            raise SystemExit(f"{label}: no records file {path}")
        fr, meta = load_records(path, det, dt)
        frames, sig = meta.get("frames"), meta.get("gt_signature")
        return label, path, fr, ((None, frames, len(frames) if frames is not None else None, sig) if frames is not None or sig else None), None, False
    info = bev_suite.legacy_record(arg, tag, n)
    if info is None:
        raise SystemExit(f"{arg} has no eval/scene_records{tag}*.pkl: evaluate it with `bev_suite.py {arg} eval --n N`")
    path = os.path.join(arg, "eval", f"scene_records{info[0]}.pkl")
    return os.path.basename(arg.rstrip("/\\")), path, load_records(path, det, dt)[0], info, bev_suite.train_gt(arg), True


def _rel(path):
    try:
        return os.path.relpath(path)
    except ValueError:                                       # another drive (Windows)
        return path


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="front", description="front-area detection comparison of several systems (first = reference)")
    p.add_argument("systems", nargs="+", help="REF first; each a run folder or LABEL=FILE.pkl")
    p.add_argument("--x-max", type=float, default=50.0, help="the sector ends here (m, at most 60); default 50")
    p.add_argument("--n", default="all", help="the n of the evaluations (scene_records<tag>_n<N>.pkl), as in `compare`; default all")
    p.add_argument("--tag", default="", help="'' cold val (default), _warm, _train ...: the suffix of the evaluation, as in `compare`")
    p.add_argument("--det", default="det", help="box set of every system: det | detr")
    p.add_argument("--dt", type=float, default=0.0)
    p.add_argument("--boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="write the results file here (markdown); default: print only")
    p.add_argument("--force", action="store_true", help="compare even when the recorded GT, the frame sets or the GT box counts differ")
    c = p.parse_args(argv)
    if len(c.systems) < 2:
        p.error("at least two systems: the reference and one more")
    from evaluation import bev_suite
    ids = [_ident(a, c.tag, c.n, c.det, c.dt) for a in c.systems]
    rl, _, _, ri, rtr, rrun = ids[0]
    notes = []
    for lab, _, _, info, tr, isrun in ids[1:]:
        if info is None or ri is None:
            notes.append(f"WARNING: {lab if info is None else rl} has no recorded GT identity or frame set (no run folder, no meta in the pickle): scenes, frames per scene "
                         "and ground-truth box counts are compared, nothing more")
        else:           # a refusal (SystemExit), or a mismatch under --force; "not recorded" is summed up below, once for all pairs
            notes += [w for w in bev_suite.gt_check(rl, lab, {"evaluation GT": (ri[3], info[3]), "training GT": (rtr, tr)}, c.force) if "not recorded" not in w]
            if (rrun and isrun) or (ri[1] is not None and info[1] is not None):
                notes += bev_suite.frames_check(rl, lab, ri, info, c.n, c.force)
    for what, missing in (("evaluation", [i[0] for i in ids if i[3] is not None and i[3][3] is None]), ("training", [i[0] for i in ids if i[5] and i[4] is None])):
        if missing:
            notes.append(f"WARNING: {what} GT signature not recorded for {', '.join(missing)}: the GT identity is unchecked for them")
    notes = list(dict.fromkeys(notes))
    for n in notes:
        print(n)
    sources = [f"`{lab}`: {_rel(path)} (evaluation GT {(info[3] if info is not None else None) or 'not recorded'})" for lab, path, _, info, _, _ in ids]
    records = f"{'train' if '_train' in c.tag else 'val'} frames, " + ("memory carried along the scenes (warm)" if "_warm" in c.tag else "every frame from an empty memory (cold)") + (f", tag '{c.tag}'" if c.tag else "")
    try:
        res = compare([(lab, fr) for lab, _, fr, *_ in ids], c.x_max, c.boot, c.seed, strict_gt=not c.force)
    except ValueError as e:
        raise SystemExit(f"front refused: {e}")
    text = markdown(res, sources, notes, records=records)
    print(text)
    if c.out:
        if os.path.dirname(c.out):
            os.makedirs(os.path.dirname(c.out), exist_ok=True)
        with open(c.out, "w", encoding="utf-8") as f:
            f.write(text)
    return 0


# ----------------------------------------------------------------------------------------------- self-test
def _world(shift=0.0, ped_shift=None, score_seed=None, round_scores=False, frames=4, scenes=3, ped_scenes=None, drop_ped=False, odd_shift=0.0, far_shift=0.0):
    """({scene: [(dets, gts)]}) of a synthetic world: per frame 5 vehicles (3 inside the 50 m sector: x 10, 25, 45; one at x 70 and one at (12, 30), 68 deg
    off the axis, outside) and 3 pedestrians (inside: x 8, 30, 44; only in the first `ped_scenes` scenes when given; `drop_ped`: without the one at x 8).
    dets = the GT boxes moved `shift` m forward (pedestrians `ped_shift`, default the same; odd frames `odd_shift` more, vehicles at x >= 40 `far_shift`
    more), score 0.9, or random (round_scores: to 0.1 steps = many ties)."""
    rng = np.random.RandomState(score_seed) if score_seed is not None else None
    ped_shift = shift if ped_shift is None else ped_shift
    veh = [(10.0, 0.0), (25.0, 3.0), (45.0, -5.0), (70.0, 0.0), (12.0, 30.0)]
    ped = [(8.0, 2.0), (30.0, -4.0), (44.0, 3.0)][1 if drop_ped else 0:]
    out = {}
    for si in range(scenes):
        fr = []
        for fi in range(frames):
            gts = [{"group": "vehicle", "cx": x, "cy": y, "length": 4.5, "width": 2.0, "yaw": 0.0, "vx": 0.0, "vy": 0.0} for x, y in veh]
            if ped_scenes is None or si < ped_scenes:
                gts += [{"group": "pedestrian", "cx": x, "cy": y, "radius": 0.4, "vx": 0.0, "vy": 0.0} for x, y in ped]
            dets = []
            for g in gts:
                sc = 0.9 if rng is None else (round(float(rng.rand()), 1) if round_scores else float(rng.rand()))
                mv = (shift if g["group"] == "vehicle" else ped_shift) + (odd_shift if fi % 2 else 0.0) + (far_shift if g["group"] == "vehicle" and g["cx"] >= 40 else 0.0)
                dets.append(dict(g, cx=g["cx"] + mv, score=sc))
            fr.append((dets, gts))
        out[f"scene-{si:04d}"] = fr
    return out


def _selftest():
    import tempfile

    from Evaluation.stats import percentile_interval, scene_picks

    def direct(world, ix, names, x_max=50.0):
        mm = DetMeter(classes=list(CLASSES), x_max=x_max)
        for i in ix:
            for ds, gs in world[names[i]]:
                mm.add(ds, gs, dt=0.0)
        return mm.result()[0.0]

    def direct_map2(world, ix, names, x_max=50.0):
        return map2(direct(world, ix, names, x_max))

    A, B = _world(), _world(shift=3.0)
    res = compare([("perfect", A), ("shift3", B)], x_max=50.0, n_boot=50)
    p, s = res["systems"]
    assert abs(p["map2"] - 1.0) < 1e-9, p["map2"]                          # every in-sector box found at every threshold
    assert abs(s["map2"] - 0.25) < 1e-9, s["map2"]                         # 3 m off: only the 4 m threshold finds it (AP 1), the others 0
    assert abs(s["ap"]["vehicle"][4.0] - 1.0) < 1e-9 and s["ap"]["vehicle"][2.0] == 0.0, s["ap"]
    assert abs(s["diff"] + 0.75) < 1e-9 and abs(s["lo"] + 0.75) < 1e-9 and abs(s["hi"] + 0.75) < 1e-9 and s["n_valid"] == 50, s     # constant difference
    assert res["n_gt"] == {"vehicle": 3 * 12, "pedestrian": 3 * 12}, res["n_gt"]              # x 70 and the 68-degree box are outside; 12 frames
    assert list(p["bins"]) == ["0-20", "20-40", "40-60"] and abs(p["bins"]["40-60"] - 1.0) < 1e-9 and abs(s["bins"]["20-40"] - 0.25) < 1e-9, p["bins"]
    assert "| system | 0-20 m | 20-40 m | 40-50 m |" in markdown(res), "the last bin ends at the sector (x_max 50)"
    r20 = compare([("perfect", A), ("shift3", B)], x_max=20.0, n_boot=10)
    assert list(r20["systems"][0]["bins"]) == ["0-20"] and r20["n_gt"] == {"vehicle": 12, "pedestrian": 12}, r20["n_gt"]     # x 10 / 8 only

    # the classes are averaged, not mixed: vehicles 3 m off, pedestrians exact -> (0.25 + 1.0) / 2
    V = compare([("perfect", A), ("veh3", _world(shift=3.0, ped_shift=0.0))], n_boot=5)["systems"][1]
    assert abs(V["map2"] - 0.625) < 1e-9 and V["ap"]["vehicle"][2.0] == 0.0 and abs(V["ap"]["pedestrian"][2.0] - 1.0) < 1e-9, V

    # the same numbers as the direct meter (the path of the former temp/t4 script) with many equal scores (ties), and every one of the resamples
    names, n_draw = sorted(A), 40
    N = _world(shift=0.6, score_seed=3, round_scores=True)
    C = dict(_world(), **{"scene-0000": _world(shift=3.0)["scene-0000"]})            # only scene 0 is shifted: the scenes differ, so the draws differ
    E = _world(odd_shift=3.0)                                                       # the frames of a scene differ and all scores tie: the frame order matters
    G = _world(far_shift=3.0)                                                       # the far vehicle is 3 m off: the range bins differ from the headline
    out = compare([("a", A), ("noisy", N), ("c", C), ("e", E), ("g", G)], x_max=50.0, n_boot=n_draw)
    picks = list(scene_picks(len(names), n_draw, 0))
    for sys_, world in zip(out["systems"], (A, N, C, E, G)):
        assert abs(sys_["map2"] - direct_map2(world, range(len(names)), names)) < 1e-12, sys_["label"]
        want = [direct_map2(world, ix, names) for ix in picks]
        assert np.allclose(sys_["draws"], want, atol=1e-12, equal_nan=True), sys_["label"]          # every draw, in order
        full = direct(world, range(len(names)), names)
        assert all(abs(sys_["bins"][b] - map2(full, CLASSES, b, need_all=False)) < 1e-12 for b in sys_["bins"]), sys_["label"]      # and every range bin
    g = out["systems"][4]
    assert abs(g["bins"]["0-20"] - 1.0) < 1e-9 and abs(g["bins"]["40-60"] - 0.625) < 1e-9 and abs(g["map2"] - g["bins"]["0-20"]) > 0.02 and abs(g["map2"] - g["bins"]["40-60"]) > 0.02, (g["map2"], g["bins"])     # bins differ from the headline
    assert out["systems"][3]["map2"] not in (out["systems"][0]["map2"], out["systems"][1]["map2"])
    assert len(set(round(x, 9) for x in out["systems"][2]["draws"])) > 3 and out["systems"][2]["lo"] < out["systems"][2]["hi"]
    d = [a_ - b_ for a_, b_ in zip(out["systems"][2]["draws"], out["systems"][0]["draws"])]
    assert (out["systems"][2]["lo"], out["systems"][2]["hi"])[0] == percentile_interval(d)[0] and out["systems"][2]["n_valid"] == n_draw
    assert compare([("a", A), ("c", C)], n_boot=n_draw, seed=1)["systems"][1]["draws"] != out["systems"][2]["draws"]               # the seed matters

    # a class with ground truth in some scenes only: some draws have nothing to score (counted), the bins show the classes that are there
    P = _world(ped_scenes=1)
    res_p = compare([("a", P), ("b", _world(shift=3.0, ped_scenes=1))], n_boot=60)
    assert 0 < res_p["systems"][1]["n_valid"] < 60 and "draws scorable" in markdown(res_p), res_p["systems"][1]["n_valid"]
    nop = compare([("a", _world(ped_scenes=0)), ("b", _world(ped_scenes=0))], n_boot=3)
    assert nop["systems"][0]["map2"] != nop["systems"][0]["map2"] and abs(nop["systems"][0]["bins"]["0-20"] - 1.0) < 1e-9 and "0-20" in nop["systems"][0]["partial"]
    assert "1.000*" in markdown(nop) and "no ground truth in the bin; *: only some" in markdown(nop)

    # refusals: different scenes, frame counts, ground-truth box counts (not the same GT), an empty record, a sector beyond the bins
    for bad in (dict(a=A, b={"scene-0000": A["scene-0000"]}), dict(a=A, b=_world(frames=3))):
        try:
            compare([("a", bad["a"]), ("b", bad["b"])], n_boot=2)
            raise AssertionError("different scenes / frame counts must be refused")
        except ValueError:
            pass
    try:
        compare([("a", A), ("b", _world(drop_ped=True))], n_boot=2)
        raise AssertionError("another number of ground-truth boxes must be refused")
    except ValueError as e:
        assert "ground-truth boxes" in str(e), e
    forced = compare([("a", A), ("b", _world(drop_ped=True))], n_boot=2, strict_gt=False)
    assert forced["warnings"] and "WARNING (--force)" in markdown(forced)
    for kw in (dict(systems=[("a", {}), ("b", {})]), dict(systems=[("a", A), ("b", A)], x_max=70.0)):
        try:
            compare(n_boot=2, **kw)
            raise AssertionError("an empty record / a sector beyond the range bins must be refused")
        except ValueError:
            pass
    text = markdown(res, ["`perfect`: demo"], ["a note"], records="val frames, cold")
    assert "| shift3 | 0.250 | -0.750 [-0.750, -0.750] |" in text and "GT boxes in the area (perfect): vehicle 36, pedestrian 36." in text and "val frames, cold, 12 frames" in text, text

    # the command line on run folders and pickles
    with tempfile.TemporaryDirectory() as tmp:
        def make(name, world, sig, train_sig=None, frames=True, meta=True, tag="_n12"):
            os.makedirs(os.path.join(tmp, name, "eval"), exist_ok=True)
            keys = [f"{s}/{i}" for s, fr in sorted(world.items()) for i in range(len(fr))] if frames else None
            m = dict(({"frames": keys} if frames else {}), gt_signature=sig) if meta else {}
            with open(os.path.join(tmp, name, "eval", f"scene_records{tag}.pkl"), "wb") as f:
                pickle.dump({"seg": {}, "det": {s: {0.0: fr} for s, fr in world.items()}, "detr": {}, "meta": m}, f)
            if train_sig:
                with open(os.path.join(tmp, name, "gt.json"), "w") as f:
                    f.write('{"gt_signature": "%s"}' % train_sig)
            return os.path.join(tmp, name)

        def refused(*argv):
            try:
                main(list(argv))
            except SystemExit as e:
                return str(e)
            return None
        a, b = make("a", A, "G1", "T1"), make("b", B, "G1", "T1")
        c_, t2 = make("c", B, "G2", "T1"), make("t2", B, "G1", "T2")
        w = make("w", B, "G1", "T1", tag="_warm_n12")
        shuffled = make("fr", B, "G1", "T1")                                              # same GT, other frame keys
        with open(os.path.join(shuffled, "eval", "scene_records_n12.pkl"), "rb") as f:
            rec = pickle.load(f)
        rec["meta"]["frames"] = [k + "x" for k in rec["meta"]["frames"]]
        with open(os.path.join(shuffled, "eval", "scene_records_n12.pkl"), "wb") as f:
            pickle.dump(rec, f)
        out_md = os.path.join(tmp, "sub", "out.md")
        common = ["--n", "12", "--boot", "5"]
        assert "different GT" in (refused(a, c_, *common) or ""), "a different recorded evaluation GT must be refused"
        assert "different GT" in (refused(a, t2, *common) or ""), "a different recorded training GT must be refused"
        assert "frame" in (refused(a, shuffled, *common) or ""), "different frame keys must be refused"
        assert refused(a, c_, *common, "--force") is None and refused(a, shuffled, *common, "--force") is None
        assert main([a, b, *common, "--out", out_md]) == 0                                  # --out creates the folder
        txt = open(out_md, encoding="utf-8").read()
        assert "| b | 0.250 |" in txt and "evaluation GT G1" in txt and "val frames, every frame from an empty memory (cold)" in txt, txt
        w2 = make("w2", A, "G1", "T1", tag="_warm_n12")
        assert main([w2, w, "--n", "12", "--tag", "_warm", "--boot", "5", "--out", out_md]) == 0                    # the warm records: said in the header
        assert "val frames, memory carried along the scenes (warm), tag '_warm'" in open(out_md, encoding="utf-8").read()
        assert "no eval/scene_records_warm" in (refused(a, w, "--n", "12", "--tag", "_warm") or "").replace("scene_records_warm*", "scene_records_warm"), "a run without the warm records is named"
        ext_ok = "ext=" + os.path.join(b, "eval", "scene_records_n12.pkl")                  # a pickle with meta: its signature is checked
        assert main([a, ext_ok, *common]) == 0
        bad_ext = make("e2", B, "G2")
        assert "different GT" in (refused(a, "ext=" + os.path.join(bad_ext, "eval", "scene_records_n12.pkl"), *common) or ""), "a pickle with another GT signature must be refused"
        bare = make("e3", B, None, meta=False)
        assert main([a, "ext=" + os.path.join(bare, "eval", "scene_records_n12.pkl"), *common]) == 0       # no meta at all: warned, scenes / counts compared
        drop = make("e4", _world(drop_ped=True), None, meta=False)
        assert "ground-truth boxes" in (refused(a, "ext=" + os.path.join(drop, "eval", "scene_records_n12.pkl"), *common) or ""), "another GT box count must be refused without meta too"
        assert refused(a, "ext=" + os.path.join(drop, "eval", "scene_records_n12.pkl"), *common, "--force", "--out", out_md) is None      # --force: a warning in the file
        assert "WARNING (--force): ext has" in open(out_md, encoding="utf-8").read()
        assert "no records file" in (refused(a, "ext=" + os.path.join(tmp, "missing.pkl"), *common) or "")
        assert "empty record" in (refused(a, b, *common, "--det", "detr") or ""), "the empty detr set must be refused"
        assert "beyond the last range bin" in (refused(a, b, *common, "--x-max", "70") or "")
    print("front_compare selftest ok: known answers (perfect 1.0, 3 m shift 0.25, classes averaged, sector and 68-degree boxes dropped), every resample equal to the "
          "direct DetMeter with ties, one class missing, scenes / frame counts / GT box counts / recorded GT (run and pickle) / empty record / wide sector refused, command line")
    return 0


if __name__ == "__main__":
    raise SystemExit(_selftest() if len(sys.argv) == 1 else main())
