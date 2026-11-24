"""crn_convert - CRN's detections (nuScenes json, global frame) -> the front_compare record of OUR frames (card t26, G0, the object row of the SOA table).

    python evaluation/crn/crn_convert.py RESULTS_JSON OUT.pkl [--info INFOS.pkl] [--tables DIR] [--min-score 0.05] [--expect-signature SIG] [--partial]

RESULTS_JSON = `results_nusc.json` of `crn_export.py` (CRN's own writer: boxes of the key frame in the GLOBAL frame at the annotation (lidar) time, size w l h, quaternion
w x y z, velocity in the global axes). OUT.pkl = what `bev_suite.py front REF ... LABEL=OUT.pkl` reads (the format of the scratch script soa_front_v8.py, DualBEV, removed with the scratch folder in D140; a sample is runs/t16_soa_ref/dualbev_front_v8_n30.pkl):
{"seg": {}, "det": {scene: {0.0: [(dets, gt boxes) per frame]}}, "detr": {scene: {}}, "meta": {"frames": [tokens], "gt_signature": ...}}.

The rule, written once (the helpers below, each one line of it):
  frames   the samples of the CRN info file that lie in our VAL scenes (splits/xl.json) and have a previous and a next key frame (the dataset's own rule): 1620 of the 1706
  GT       the stored front GT of the LIVE generator (data/autogt/<signature>/): refused when the folder, its index.json and the live signature differ, or when --expect-signature
           names another; boxes by evaluation.metrics.gt_boxes_from_meta, as the evaluation suite keeps them; the frame of the GT = the ego of the CAM_FRONT time, meta["T_e2g"]
  time     the GT moves its boxes by v * (t_camera - t_annotation) (GtGenerator/nuscenes_gt.py, "boxes are at the camera time"); a CRN box is at the annotation time, so it is
           moved the same way by ITS velocity (always: a detector has no "stationary" flag); t_camera - t_annotation = meta["timestamp"] - meta["annotation_timestamp"]
  frame    centre = R^T (p - t) with T_e2g = [R | t]; velocity = R[:2,:2]^T v; yaw = wrap(yaw_global - heading of T_e2g), the yaw of a quaternion as the devkit's yaw_pitch_roll[0]
  classes  car, truck, bus, trailer, construction_vehicle -> vehicle; motorcycle, bicycle -> two_wheeler; pedestrian -> pedestrian (the groups of the DualBEV export); barrier and
           traffic_cone dropped. A det has the keys of training/legacy/dualbev_front.py decode (group, cls, score, cx, cy, vx, vy, length, width, yaw)
  score    only boxes with score >= --min-score (the DualBEV export's 0.05; CRN's own decode keeps 0.01), nothing else is cut (no top-k: CRN sees 360 degrees)
Cross-checks that stop the run: the info's timestamps equal the GT's (annotation and CAM_FRONT time), a missing token in the json (unless --partial), a non-finite box.
Self-test and the planted-bug test: tests/test_crn_export.py (CPU).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
WS = os.path.abspath(os.path.join(ROOT, ".."))
for _p in (ROOT, WS, os.path.join(WS, "autoresearch")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

CLASS_GROUP = {"car": "vehicle", "truck": "vehicle", "bus": "vehicle", "trailer": "vehicle", "construction_vehicle": "vehicle",
               "motorcycle": "two_wheeler", "bicycle": "two_wheeler", "pedestrian": "pedestrian"}
DROPPED = ("barrier", "traffic_cone")
HOME = os.path.expanduser("~")
INFOS = f"{HOME}/data/nuscenes_full/nuscenes_infos_val_crn.pkl"
TABLES = f"{HOME}/data/nuscenes_full/v1.0-trainval"
SPLIT = os.path.join(ROOT, "splits", "xl.json")             # the VAL scenes (the same 82 for large and xl)


# ------------------------------------------------------------------------------------------------ the rule, one helper per line of it
def wrap(angle):
    return math.remainder(angle, 2 * math.pi)


def parse_quat(q):
    """json rotation [w, x, y, z] -> unit quaternion (w, x, y, z); CRN's writer does not normalise (the mean of six camera poses is not a unit quaternion)"""
    q = np.asarray(q, float)
    return q / np.linalg.norm(q)


def quat_yaw(q):
    """yaw of a unit quaternion (w, x, y, z): the devkit's Quaternion.yaw_pitch_roll[0], which the GT uses for the annotations"""
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def time_gap(meta):
    """s from the annotation time to the camera time as the GT applies it: negative (about -0.036 s), 0 when the GT keeps the annotation time"""
    return (meta["timestamp"] - meta["annotation_timestamp"]) * 1e-6 if meta.get("boxes_at") == "camera" else 0.0


def shift_to_camera_time(p, v, dt):
    """world position at the camera time: constant velocity over dt, z unchanged (nuscenes_gt._boxes)"""
    return p + np.array([v[0] * dt, v[1] * dt, 0.0])


def to_ego_point(T_e2g, p):
    return T_e2g[:3, :3].T @ (p - T_e2g[:3, 3])


def to_ego_vector(T_e2g, v):
    return T_e2g[:2, :2].T @ v


def to_ego_yaw(T_e2g, yaw):
    return wrap(yaw - math.atan2(T_e2g[1, 0], T_e2g[0, 0]))


def footprint(size):
    """nuScenes size [w, l, h] -> (length, width)"""
    return float(size[1]), float(size[0])


def group_of(name):
    """class name -> group; None for the dropped classes; an unknown name is an error (a silent drop would hide a wrong class list)"""
    if name in DROPPED:
        return None
    return CLASS_GROUP[name]


def box_to_ego(anno, T_e2g, dt):
    """one CRN box of the json (global frame, annotation time) -> a det dict of our ego frame at the CAM_FRONT time, or None for a dropped class"""
    group = group_of(anno["detection_name"])
    if group is None:
        return None
    v = np.asarray(anno["velocity"][:2], float)
    p = shift_to_camera_time(np.asarray(anno["translation"], float), v, dt)
    c, ve = to_ego_point(T_e2g, p), to_ego_vector(T_e2g, v)
    length, width = footprint(anno["size"])
    out = {"group": group, "cls": anno["detection_name"], "score": float(anno["detection_score"]), "cx": float(c[0]), "cy": float(c[1]), "vx": float(ve[0]), "vy": float(ve[1]),
           "length": length, "width": width, "yaw": to_ego_yaw(T_e2g, quat_yaw(parse_quat(anno["rotation"])))}
    if not all(math.isfinite(out[k]) for k in ("score", "cx", "cy", "vx", "vy", "length", "width", "yaw")):
        raise ValueError(f"a non-finite box: {anno}")
    return out


# ------------------------------------------------------------------------------------------------ frames, GT, identity
def val_samples(infos, prev_next, val_scenes):
    """[{scene, token, ts, cam_ts}] of the info samples in the VAL scenes with a previous and a next key frame, in the order of the file (scene by scene, time order);
    prev_next = {token: (prev, next)} of nuScenes sample.json"""
    out = []
    for i in infos:
        if i["scene_name"] in val_scenes and all(prev_next[i["sample_token"]]):
            out.append({"scene": i["scene_name"], "token": i["sample_token"], "ts": i["timestamp"], "cam_ts": i["cam_infos"]["CAM_FRONT"]["timestamp"]})
    return out


def prev_next_of(tables=TABLES):
    with open(os.path.join(tables, "sample.json")) as f:
        return {s["token"]: (s["prev"], s["next"]) for s in json.load(f)}


def gt_store(expect=None):
    """-> (store folder, GT signature string as the evaluation records it '<res>_<ss>_<signature>', version) of the LIVE generator's stored front GT; refuses a mismatch"""
    from GtGenerator import current_signature, versions
    sig = current_signature()
    store = os.path.join(WS, "data", "autogt", sig)
    try:
        with open(os.path.join(store, "index.json")) as f:
            idx = json.load(f)
    except OSError:
        raise SystemExit(f"no stored GT {store} for the live signature {sig}")
    if idx.get("signature") != sig:
        raise SystemExit(f"{store}/index.json says signature {idx.get('signature')}, the live generator is {sig}")
    if expect and versions.canonical(expect.rsplit("_", 1)[-1]) != sig:
        raise SystemExit(f"--expect-signature {expect} is not the live GT {sig}")
    return store, f"{idx['res']:g}_{idx['ss']}_{sig}", versions.version_of_key(sig)


def read_meta(store, token):
    """the meta of one stored keyframe without its layers (an npz member is read when it is asked for)"""
    with np.load(os.path.join(store, token + ".npz"), allow_pickle=True) as z:
        return z["meta"].item()


def check_times(meta, frame):
    """the info's timestamps (CRN's) against the GT's: the annotation time and the CAM_FRONT time must be the ones the GT moved its boxes between"""
    if meta["sample"] != frame["token"] or meta["annotation_timestamp"] != frame["ts"] or meta["timestamp"] != frame["cam_ts"]:
        raise SystemExit(f"{frame['token']}: GT times {meta['annotation_timestamp']} / {meta['timestamp']} (annotation / camera) differ from the info's {frame['ts']} / {frame['cam_ts']}")


def convert(results, frames, store, signature, min_score=0.05):
    """results = {token: [annos]} of the json; frames = val_samples(...) -> the front_compare record"""
    from evaluation.metrics import gt_boxes_from_meta
    det, n_det = {}, 0
    for fr in frames:
        meta = read_meta(store, fr["token"])
        check_times(meta, fr)
        T, dt = np.asarray(meta["T_e2g"], float), time_gap(meta)
        dets = [d for a in results[fr["token"]] if a["detection_score"] >= min_score for d in [box_to_ego(a, T, dt)] if d is not None]
        n_det += len(dets)
        det.setdefault(fr["scene"], {0.0: []})[0.0].append((dets, gt_boxes_from_meta(meta["boxes"])))
    return {"seg": {}, "det": det, "detr": {s: {} for s in det},
            "meta": {"frames": [f["token"] for f in frames], "gt_signature": signature, "producer": "evaluation/crn/crn_convert.py", "min_score": min_score,
                     "n_dets": n_det, "rule": "global -> ego of CAM_FRONT time (GT T_e2g), boxes moved by v * (t_camera - t_annotation), classes merged to vehicle / two_wheeler / pedestrian"}}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("results", help="results_nusc.json of crn_export.py")
    p.add_argument("out", help="the front_compare pickle to write")
    p.add_argument("--info", default=INFOS)
    p.add_argument("--tables", default=TABLES, help="the nuScenes json tables (sample.json: previous / next)")
    p.add_argument("--min-score", type=float, default=0.05)
    p.add_argument("--expect-signature", default=None, help="the GT signature the card names (a107e0b4e8 or 0.5_4_a107e0b4e8); refused when it is not the live one")
    p.add_argument("--partial", action="store_true", help="convert only the frames that the json holds (a smoke run); default: a missing frame is an error")
    a = p.parse_args(argv)
    store, signature, version = gt_store(a.expect_signature)
    print(f"GT: {version}, signature {signature}, store {os.path.relpath(store, WS)}", flush=True)
    with open(a.info, "rb") as f:
        infos = pickle.load(f)
    with open(SPLIT) as f:
        val = set(json.load(f)["val"])
    frames = val_samples(infos, prev_next_of(a.tables), val)
    print(f"{len(infos)} CRN samples in {len({i['scene_name'] for i in infos})} scenes; {len(frames)} are VAL samples with a previous and a next key frame "
          f"({len({f['scene'] for f in frames})} scenes)", flush=True)
    with open(a.results) as f:
        results = json.load(f)["results"]
    missing = [fr["token"] for fr in frames if fr["token"] not in results]
    if missing and not a.partial:
        raise SystemExit(f"{len(missing)} of {len(frames)} frames are not in {a.results} (first {missing[0]}); --partial converts the others")
    frames = [fr for fr in frames if fr["token"] in results]
    rec = convert(results, frames, store, signature, a.min_score)
    tmp = a.out + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(rec, f)
    os.replace(tmp, a.out)
    print(f"{len(frames)} frames of {len(rec['det'])} scenes, {rec['meta']['n_dets']} CRN boxes with score >= {a.min_score:g} -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
