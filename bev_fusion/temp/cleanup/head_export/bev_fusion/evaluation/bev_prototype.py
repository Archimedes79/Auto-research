"""The ready-to-use prototype of a run: planner-ready outputs of chosen val scenes (decision D28).

    python evaluation/bev_prototype.py RUN [--scenes scene-0003 5 ...] [--frames N] [--out DIR] [--video] [--device cuda|cpu]

RUN = runs/<name>. Every chosen scene is run frame by frame in time order with the chain's memory carried along it (the suite's warm
mode, as deployed): the network runs once per frame (bev_adapter.BEVSubject: EMA weights of the last checkpoint under the run's own
code snapshot). --scenes takes val scene names or indices into the val scene list (`bev_suite.py RUN list`), default the first;
--frames N the first N frames of each scene (default all). Outputs in --out (default RUN/prototype/), the fields in
evaluation/README.md "Prototype outputs":
    objects.json          per scene and frame: the boxes in the ego frame (class, centre, size, yaw, velocity, score) at dt 0 and the +0.5 s forecast
    maps_<scene>.npz      per frame the BEV probability map of every layer at dt 0, with the grid (resolution, extent, axes)
    summary.json / .md    run, checkpoint, GT signatures, frames, and the headline numbers COPIED from the run's stored eval/card.json
                          (keys: evaluation/board.py COLUMNS) and the ms per frame of eval/latency.json; nothing is recomputed here
    qualitative/videos/   with --video: the suite's dashboard video of each scene (`qual videos`; it runs the scene a second time)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "..", "autoresearch"))
sys.path.insert(0, ROOT)

FRAME = ("ego (vehicle) frame of the frame's own time, metres: x forward, y left; yaw in rad counter-clockwise from x; velocity = world "
         "velocity of the object in the ego axes, m/s (the autogt GT definition)")
FORECAST = ("+0.5 s: the boxes at the next keyframe (t + 0.40 ... 0.60 s) in the ego frame of that time; decoded, as in the evaluation, "
            "with the true gap and ego motion of that keyframe (from the GT: a deployed system supplies its own ego motion)")


def pick_scenes(source, specs):
    """val scene names or indices into source.scenes(); none = the first scene"""
    names = source.scenes()
    out = []
    for s in specs or ["0"]:
        name = names[int(s)] if str(s).isdigit() else s
        if name not in names:
            raise SystemExit(f"{s}: not a scene of this split (see `bev_suite.py RUN list`)")
        out.append(name)
    return out


def at_dt(outputs, dt):
    """the subject's output at time dt (keys are the nominal dts as floats), None when it is not decoded"""
    return next((o for k, o in outputs["dt"].items() if abs(float(k) - dt) < 1e-6), None)


def boxes_of(o):
    """(label, boxes) of one dt's output: the refined boxes when the model has a refiner, else the object head's decoded ones"""
    return ("detr", o["detr"]) if o.get("detr") is not None else ("det", o["dets"])


def planner_object(d: dict) -> dict:
    """one box of object_head.decode (or the refiner) -> the planner record: boxes have size [length, width] and yaw, circles
    (pedestrians) a radius and no yaw"""
    rec = {"class": d["group"], "score": float(d["score"]), "centre": [float(d["cx"]), float(d["cy"])],
           "velocity": [float(d["vx"]), float(d["vy"])]}
    if "radius" in d:
        rec.update(radius=float(d["radius"]), size=None, yaw=None)
    else:
        rec.update(radius=None, size=[float(d["length"]), float(d["width"])], yaw=float(d["yaw"]))
    return rec


def grid_of(grid) -> dict:
    return {"res": float(grid.res), "x_min": float(grid.x_min), "x_max": float(grid.x_max), "y_min": float(grid.y_min),
            "y_max": float(grid.y_max), "nx": int(grid.nx), "ny": int(grid.ny),
            "axes": "frame, x, y: map[f, ix, iy] is the cell centred at x = x_min + (ix + 0.5) res (forward), y = y_min + (iy + 0.5) res (left)"}


def export_scene(suite, scene: str, n_frames, out: str):
    """run one scene with the memory carried; write maps_<scene>.npz; -> (objects of each frame, predict ms of each frame)"""
    from evalsuite.runner import Runner
    refs = suite.source.frames(scene)[: n_frames or None]
    frames, maps, ms = [], {}, []
    for sample, pred in Runner(suite.source, suite.subject, "carry").iter_scene(scene, refs):
        o0, o5 = at_dt(pred.outputs, 0.0), at_dt(pred.outputs, 0.5)
        label, boxes = boxes_of(o0)
        rec = {"index": sample.ref.index, "token": sample.ref.key, "timestamp_us": sample.meta.get("timestamp"), "box_source": label,
               "objects": [planner_object(d) for d in boxes],
               "forecast_0.5s": None if o5 is None else [planner_object(d) for d in boxes_of(o5)[1]]}
        frames.append(rec)
        for name, p in o0["prob"].items():
            maps.setdefault(name, []).append(np.asarray(p, np.float32))
        ms.append(pred.info["predict_ms"])
    g = grid_of(suite.subject.model.grid)
    np.savez_compressed(os.path.join(out, f"maps_{scene}.npz"), **{k: np.stack(v) for k, v in maps.items()},
                        layers=np.array(sorted(maps)), index=np.array([f["index"] for f in frames]),
                        token=np.array([f["token"] for f in frames]), grid=np.array(json.dumps(g)))
    return frames, ms


def stored_numbers(run: str) -> dict:
    """the headline numbers as stored by `bev_suite.py RUN eval` (card.json, its frames and memory mode from metrics.json) and the
    latency of `bev_suite.py RUN time` (latency.json); never recomputed"""
    from evaluation.board import COLUMNS
    from evaluation.bev_suite import _read_json
    card_path, met_path, lat_path = (os.path.join(run, "eval", f) for f in ("card.json", "metrics.json", "latency.json"))
    card, met, lat = _read_json(card_path), _read_json(met_path) or {}, _read_json(lat_path)
    head = None
    if card:
        head = {"source": os.path.relpath(card_path, ROOT), "split": card.get("split"), "frames": card.get("frames"),
                "memory": met.get("memory"), "suite_run": met.get("suite_run"), "date": card.get("date"),
                "gt_signature": (card.get("protocol") or {}).get("gt_signature"),
                "numbers": [{"group": g, "label": lab, "key": k, "better": b, "value": card["metrics"].get(k)} for g, lab, k, b in COLUMNS]}
    ms = "not measured"
    if lat:
        L = lat.get("latency") or {}
        ms = {"source": os.path.relpath(lat_path, ROOT), "gpu": lat.get("gpu"), "idle": L.get("idle"), "median_ms": L.get("median_ms"),
              "p90_ms": L.get("p90_ms"), "camera_heads_median_ms": L.get("camera_heads_median"),
              "note": "the cached chain at batch 1, dt -0.5 / 0 / +0.5, without the pre-trained camera heads"}
    return {"headline": head, "ms_per_frame": ms}


def _num(v) -> str:
    return "-" if v is None else f"{v:.4f}"


def summary_md(s: dict) -> str:
    lines = [f"# prototype of {s['run']}", "", f"checkpoint `{s['checkpoint']}` (step {s['step']}, EMA weights), device {s['device']}, "
             f"memory carried along each scene", "",
             f"GT signature: now {s['gt_signature']['now']}, training {s['gt_signature']['training']}, stored card "
             f"{s['gt_signature']['card']}" + ("" if s["gt_signature"]["same"] else "  **(they differ: the stored numbers are of another GT)**"), "",
             "| scene | frames | mean predict ms (this pass) |", "|---|---|---|"]
    lines += [f"| {sc['scene']} | {sc['frames']} | {sc['predict_ms_mean']:.0f} |" for sc in s["scenes"]]
    h = s["headline"]
    lines += [""]
    if h is None:
        lines += ["headline numbers: none stored (no eval/card.json; run `bev_suite.py RUN eval --n 1000`)"]
    else:
        lines += [f"## headline numbers (copied from `{h['source']}`: {h['split']}, {h['frames']} frames, memory {h['memory']}, {h['date']})", "",
                  "| head | number | key | value | better |", "|---|---|---|---|---|"]
        lines += [f"| {r['group']} | {r['label']} | `{r['key']}` | {_num(r['value'])} | {r['better']} |"
                  for r in h["numbers"]]
    m = s["ms_per_frame"]
    lines += ["", "ms per frame: " + (m if isinstance(m, str) else f"median {m['median_ms']:.1f}, p90 {m['p90_ms']:.1f} on {m['gpu']} "
                                     f"(idle {m['idle']}; {m['note']}; from `{m['source']}`)"),
              "", "outputs: objects.json, maps_<scene>.npz" + (", qualitative/videos/" if s["videos"] else "")
              + "; fields in evaluation/README.md (Prototype outputs)", ""]
    return "\n".join(lines)


def export(suite, run: str, scenes, n_frames, out: str, video: bool = False) -> dict:
    """the prototype outputs of the chosen scenes into out; -> the summary"""
    from evaluation.bev_suite import train_gt
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    objects, rows = {}, []
    for scene in scenes:
        frames, ms = export_scene(suite, scene, n_frames, out)
        objects[scene] = frames
        rows.append({"scene": scene, "frames": len(frames), "predict_ms_mean": float(np.mean(ms)), "maps": f"maps_{scene}.npz"})
    subj = suite.subject.describe()
    with open(os.path.join(out, "objects.json"), "w") as f:
        json.dump({"run": run, "checkpoint": subj["checkpoint"], "frame": FRAME, "forecast": FORECAST, "scenes": objects}, f, indent=1)
    videos = suite.qual.videos(scenes=scenes, max_frames=n_frames or None, view="dashboard") if video else []
    s = {"run": run, "checkpoint": subj["checkpoint"], "step": subj["step"], "device": subj["device"], "code": subj["code"],
         "scenes": rows, "videos": [os.path.relpath(v["video"]["path"], out) for v in videos], "seconds": round(time.time() - t0, 1),
         "predict_ms_note": "wall time of subject.predict in this pass (all dts + box decoding, on this device and load): not the latency protocol"}
    s.update(stored_numbers(run))
    now = suite.source.describe().get("gt_signature")
    card_gt = (s["headline"] or {}).get("gt_signature")
    s["gt_signature"] = {"now": now, "training": train_gt(run), "card": card_gt, "same": now == card_gt == train_gt(run)}
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(s, f, indent=1)
    with open(os.path.join(out, "summary.md"), "w") as f:
        f.write(summary_md(s))
    return s


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run")
    p.add_argument("--scenes", nargs="*", default=None, help="val scene names or indices (default: the first val scene)")
    p.add_argument("--frames", type=int, default=0, help="the first N frames of each scene (0 = all)")
    p.add_argument("--out", default=None, help="default RUN/prototype")
    p.add_argument("--video", action="store_true", help="the suite's dashboard video of each scene")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    a = p.parse_args(argv)
    run = a.run.rstrip("/")
    out = os.path.abspath(a.out or os.path.join(run, "prototype"))
    from evalsuite import EvalSuite
    from evaluation.bev_adapter import build
    source, subject, task = build(run, device=a.device, workers=0)        # build() moves to the bev_fusion folder
    suite = EvalSuite(source, subject, task, out=out, state="carry")
    s = export(suite, run, pick_scenes(source, a.scenes), a.frames, out, a.video)
    suite.close()
    print(summary_md(s))
    print(f"-> {out} ({s['seconds']} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
