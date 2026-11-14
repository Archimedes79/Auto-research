"""Gate of the CRN export for the SOA object row (card t26, G0): `evaluation/crn/crn_convert.py` (CRN's json -> the front_compare record of our frames) and the gates of
`crn_gates.py` / `crn_devkit_gate.py`. CPU, seconds, no data, no CRN, no GPU (env gcar). Usage (from the bev_fusion folder): python tests/test_crn_export.py

1  a planted box with a known answer: known global pose and quaternion (not a unit quaternion: CRN's writer does not normalise), ego pose (90 degrees heading), velocity and
   time gap -> the ego-frame centre, yaw, velocity and size, by hand, to 1e-9; the ten CRN classes map to vehicle / two_wheeler / pedestrian / dropped; and a second pose with pitch
   and roll against the devkit's own Box.translate / Box.rotate (centre and velocity)
2  eleven planted bugs, each one patched into the module, must each break the known answer: swapped quaternion order, no quaternion normalisation, time gap of the wrong sign, no time
   shift, y flipped, transposed rotation, yaw not relative to the ego heading, velocity not rotated, width and length swapped, trailer merged into two_wheeler, barrier kept
3  the chain on a toy store: GT boxes written to npz files, a CRN json made from them by the inverse formulas, `convert` -> the boxes are the GT boxes again (a perfect CRN) and
   `front_compare` gives mAP2 1.0; the score floor, the dropped classes, the frame rule (previous and next key frame, VAL scenes), the time cross-check against the info file
4  the gates: TF32 comparison (known shift passes / fails, a lost box, other samples refused), the overlay (hit counts, a picture is written), the devkit sample filter
"""
import math
import os
import sys
import tempfile
from unittest import mock

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "evaluation", "crn"))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "..", "autoresearch"))
import crn_convert as C  # noqa: E402
import crn_devkit_gate as D  # noqa: E402
import crn_gates as G  # noqa: E402

DT = -0.036                                              # camera 36 ms before the annotation, as nuScenes CAM_FRONT
EGO_YAW = math.pi / 2
T_EGO = np.array([[0.0, -1, 0, 100.0], [1, 0, 0, 200.0], [0, 0, 1, 0.5], [0, 0, 0, 1]])       # ego x -> global y, ego y -> global -x; at (100, 200)
META = {"timestamp": 1_000_000_000 + int(DT * 1e6), "annotation_timestamp": 1_000_000_000, "boxes_at": "camera"}
CLASSES = ("car", "truck", "bus", "trailer", "construction_vehicle", "motorcycle", "bicycle", "pedestrian", "barrier", "traffic_cone")
GROUPS = dict(car="vehicle", truck="vehicle", bus="vehicle", trailer="vehicle", construction_vehicle="vehicle", motorcycle="two_wheeler", bicycle="two_wheeler",
              pedestrian="pedestrian", barrier=None, traffic_cone=None)
# the planted box in OUR ego frame at the camera time, by hand: centre (20, -3), yaw 0.4, velocity (5, 1) m/s, size w 2.0 l 4.5 h 1.6
WANT = dict(cx=20.0, cy=-3.0, yaw=0.4, vx=5.0, vy=1.0, length=4.5, width=2.0)


def planted(name="car", scale=1.02):
    """the CRN json entry of the planted box: global at the ANNOTATION time. Global at the camera time = ego (100, 200) + R(90 deg) (20, -3) = (103, 220); global velocity R (5, 1) = (-1, 5);
    the annotation is 0.036 s later, so p_ann = p_cam - v * DT = (103 - 0.036, 220 + 0.18); the global yaw is 0.4 + 90 degrees. The quaternion is not normalised (x 1.02)."""
    yaw = 0.4 + EGO_YAW
    q = scale * np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])
    return {"detection_name": name, "detection_score": 0.9, "translation": [103.0 - 0.036, 220.0 + 0.18, 1.3], "size": [2.0, 4.5, 1.6], "rotation": q.tolist(), "velocity": [-1.0, 5.0]}


def check_known_answer():
    """the planted box through box_to_ego: every number by hand (raises AssertionError)"""
    d = C.box_to_ego(planted(), T_EGO, C.time_gap(META))
    assert abs(C.time_gap(META) - DT) < 1e-12, C.time_gap(META)
    for k, v in WANT.items():
        assert abs(d[k] - v) < 1e-9, (k, d[k], v)
    assert d["group"] == "vehicle" and d["cls"] == "car" and d["score"] == 0.9, d
    for name, group in GROUPS.items():
        out = C.box_to_ego(planted(name), T_EGO, DT)
        assert (out is None) if group is None else out["group"] == group, (name, out)


def check_against_devkit():
    """a pose with pitch and roll: centre and velocity equal the devkit's Box.translate / Box.rotate on the box moved to the camera time"""
    from nuscenes.utils.data_classes import Box
    from pyquaternion import Quaternion
    q_e = Quaternion(axis=[0, 0, 1], angle=1.1) * Quaternion(axis=[0, 1, 0], angle=0.02) * Quaternion(axis=[1, 0, 0], angle=-0.015)
    t_e = np.array([412.3, -87.1, 0.9])
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = q_e.rotation_matrix, t_e
    p, v = np.array([430.0, -70.5, 1.8]), np.array([3.2, -1.7])
    q_b = Quaternion(axis=[0, 0, 1], angle=2.3)
    anno = {"detection_name": "bus", "detection_score": 0.5, "translation": p.tolist(), "size": [2.5, 11.0, 3.2], "rotation": q_b.elements.tolist(), "velocity": v.tolist()}
    d = C.box_to_ego(anno, T, DT)
    box = Box(p + np.array([v[0] * DT, v[1] * DT, 0.0]), [2.5, 11.0, 3.2], q_b, velocity=(v[0], v[1], 0.0))
    box.translate(-t_e)
    box.rotate(q_e.inverse)
    assert np.abs(np.array([d["cx"], d["cy"]]) - box.center[:2]).max() < 1e-9, (d, box.center)
    assert np.abs(np.array([d["vx"], d["vy"]]) - box.velocity[:2]).max() < 1e-9, (d, box.velocity)


def mutants():
    """name -> (module attribute, buggy replacement)"""
    real = {n: getattr(C, n) for n in ("parse_quat", "time_gap", "shift_to_camera_time", "to_ego_point", "to_ego_yaw", "to_ego_vector", "footprint", "group_of")}
    return {
        "quaternion order swapped (x y z w)": ("parse_quat", lambda q: real["parse_quat"](np.roll(np.asarray(q, float), -1))),
        "quaternion not normalised": ("parse_quat", lambda q: np.asarray(q, float)),
        "time gap of the wrong sign": ("time_gap", lambda m: -real["time_gap"](m)),
        "no time shift": ("shift_to_camera_time", lambda p, v, dt: p),
        "y flipped": ("to_ego_point", lambda T, p: real["to_ego_point"](T, p) * np.array([1.0, -1.0, 1.0])),
        "rotation transposed (ego to global instead of global to ego)": ("to_ego_point", lambda T, p: T[:3, :3] @ (p - T[:3, 3])),
        "yaw not relative to the ego heading": ("to_ego_yaw", lambda T, yaw: C.wrap(yaw)),
        "velocity not rotated": ("to_ego_vector", lambda T, v: v),
        "width and length swapped": ("footprint", lambda s: (float(s[0]), float(s[1]))),
        "trailer merged into two_wheeler": ("group_of", lambda n: "two_wheeler" if n == "trailer" else real["group_of"](n)),
        "barrier kept as a vehicle": ("group_of", lambda n: "vehicle" if n == "barrier" else real["group_of"](n)),
    }


def check_mutants():
    check_known_answer()
    for name, (attr, bad) in mutants().items():
        with mock.patch.object(C, attr, bad):
            try:
                check_known_answer()
            except AssertionError:
                continue
            raise AssertionError(f"the planted bug '{name}' was not caught by the known answer")
    return len(mutants())


# ------------------------------------------------------------------------------------------------ the chain on a toy store
def gt_box(group, c, yaw, vel, size=(2.0, 4.5, 1.6), observed=True):
    return {"group": group, "center": np.array([c[0], c[1], 0.8]), "size": list(size), "yaw": yaw, "vel": np.array(vel, float), "observed": observed}


def inverse_anno(b, name, score=0.9):
    """a CRN json entry that a perfect detector would write for the GT box b (stored in ego axes at the camera time): the formulas written the other way round"""
    c_g = T_EGO[:3, 3] + T_EGO[:3, :3] @ b["center"]
    v_g = T_EGO[:2, :2] @ b["vel"]
    yaw = b["yaw"] + EGO_YAW
    return {"detection_name": name, "detection_score": score, "translation": [c_g[0] - v_g[0] * DT, c_g[1] - v_g[1] * DT, c_g[2]], "size": b["size"],
            "rotation": [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)], "velocity": v_g.tolist()}


def toy_world(tmp):
    """2 VAL scenes of 4 samples each + 1 other scene; GT of the 2 middle samples of each VAL scene in a store folder; returns (infos, prev_next, results, frames' GT boxes)"""
    infos, prev_next, results, gts = [], {}, {}, {}
    for si, scene in enumerate(("scene-0001", "scene-0002", "scene-0900")):
        for k in range(4):
            tok = f"s{si}k{k}"
            infos.append({"scene_name": scene, "sample_token": tok, "timestamp": META["annotation_timestamp"] + k * 500_000,
                          "cam_infos": {"CAM_FRONT": {"timestamp": META["timestamp"] + k * 500_000}}})
            prev_next[tok] = ("p" if k else "", "n" if k < 3 else "")
            boxes = [gt_box("vehicle", (15.0 + 5 * k, 2.0 - si), 0.3, (4.0, 0.5)), gt_box("vehicle", (30.0, -4.0 + si), -0.2, (0.0, 0.0)), gt_box("pedestrian", (12.0, 3.0 + k), 1.0, (0.5, 1.2), (0.6, 0.6, 1.7)),
                     gt_box("two_wheeler", (20.0, -8.0), 0.1, (3.0, 0.0), (0.8, 1.9, 1.5)), gt_box("vehicle", (40.0, 1.0), 0.0, (0.0, 0.0), observed=False)]
            meta = dict(META, sample=tok, timestamp=META["timestamp"] + k * 500_000, annotation_timestamp=META["annotation_timestamp"] + k * 500_000, T_e2g=T_EGO, boxes=boxes)
            np.savez_compressed(os.path.join(tmp, tok + ".npz"), meta=np.array(meta, dtype=object))
            names = {"vehicle": "car", "pedestrian": "pedestrian", "two_wheeler": "bicycle"}
            results[tok] = [inverse_anno(b, names[b["group"]]) for b in boxes if b["observed"]]
            results[tok] += [inverse_anno(boxes[0], "barrier"), inverse_anno(boxes[0], "car", score=0.02)]          # a dropped class, and a box below the score floor
            gts[tok] = boxes
    return infos, prev_next, results


def check_chain():
    from evaluation import front_compare as FC
    with tempfile.TemporaryDirectory() as tmp:
        infos, prev_next, results = toy_world(tmp)
        frames = C.val_samples(infos, prev_next, {"scene-0001", "scene-0002"})
        assert [f["token"] for f in frames] == ["s0k1", "s0k2", "s1k1", "s1k2"], frames          # previous and next key frame, VAL scenes only, file order
        rec = C.convert(results, frames, tmp, "0.5_4_toy", min_score=0.05)
        assert rec["meta"]["frames"] == [f["token"] for f in frames] and rec["meta"]["gt_signature"] == "0.5_4_toy" and sorted(rec["det"]) == ["scene-0001", "scene-0002"]
        assert all(len(v[0.0]) == 2 for v in rec["det"].values())
        ref = {}
        for scene, lst in rec["det"].items():
            for dets, gts in lst[0.0]:
                assert len(dets) == 4, [d["cls"] for d in dets]                      # 4 observed GT boxes; the barrier and the 0.02 box are gone
                assert len(gts) == 4 and {g["group"] for g in gts} == {"vehicle", "pedestrian", "two_wheeler"}      # the unobserved box is dropped by gt_boxes_from_meta
                for d in dets:
                    g = min((g for g in gts if g["group"] == d["group"]), key=lambda g: math.hypot(g["cx"] - d["cx"], g["cy"] - d["cy"]))
                    assert math.hypot(g["cx"] - d["cx"], g["cy"] - d["cy"]) < 1e-9 and abs(g["vx"] - d["vx"]) + abs(g["vy"] - d["vy"]) < 1e-9, (d, g)
                    if "radius" not in g:
                        assert abs(g["yaw"] - d["yaw"]) < 1e-9 and abs(g["length"] - d["length"]) + abs(g["width"] - d["width"]) < 1e-9, (d, g)
            ref[scene] = [([dict(g, score=1.0) for g in gts], gts) for _, gts in lst[0.0]]
        res = FC.compare([("gt", ref), ("crn", {s: lst[0.0] for s, lst in rec["det"].items()})], x_max=50.0, n_boot=5)
        assert abs(res["systems"][0]["map2"] - 1.0) < 1e-9 and abs(res["systems"][1]["map2"] - 1.0) < 1e-9, [s["map2"] for s in res["systems"]]    # a perfect CRN scores like the GT itself
        bad = dict(frames[0], ts=frames[0]["ts"] + 1)
        try:
            C.convert(results, [bad], tmp, "0.5_4_toy")
            raise AssertionError("a timestamp that differs from the GT's must be refused")
        except SystemExit as e:
            assert "differ" in str(e), e
    if os.path.isdir(os.path.join(ROOT, "..", "data", "autogt")):                # the live GT identity needs the stored GT of this checkout
        try:
            C.gt_store(expect="deadbeef00")
            raise AssertionError("another GT signature than the live one must be refused")
        except SystemExit as e:
            assert "not the live GT" in str(e), e
        store, sig, version = C.gt_store()
        assert sig.endswith("_" + os.path.basename(store)) and version, (store, sig, version)


# ------------------------------------------------------------------------------------------------ the gates
def dump(boxes, labels, scores, tf32):
    return {"meta": {"tf32": tf32}, "samples": {"t0": {"boxes": np.array(boxes, np.float32), "labels": np.array(labels), "scores": np.array(scores, np.float32)}}}


def row(x, y, z=0.5):
    return [x, y, z, 4.5, 2.0, 1.6, 0.1, 1.0, 0.0]


def check_gates():
    import pickle
    A = dump([row(10, 0), row(20, 5), row(30, -5)], [0, 0, 8], [0.9, 0.8, 0.7], False)
    for shift, lost, verdict in ((0.05, 0, True), (0.2, 0, False)):
        B = dump([row(10 + shift, 0), row(20, 5 + shift), row(30, -5)], [0, 0, 8], [0.9, 0.8, 0.7], True)
        d, n, nl = G.compare_dumps(A["samples"], B["samples"])
        assert n == 3 and nl == lost and abs(d.max() - shift) < 1e-6 and (d.max() <= 0.1) == verdict, (d, n, nl)
    d, n, nl = G.compare_dumps(A["samples"], dump([row(10, 0), row(20, 5)], [0, 0], [0.9, 0.8], True)["samples"])
    assert nl == 1 and n == 3, (n, nl)                                                       # the pedestrian of A has no partner in B
    d, n, nl = G.compare_dumps(A["samples"], dump([row(10, 0), row(20, 5), row(30, -5)], [0, 0, 1], [0.9, 0.8, 0.7], True)["samples"])
    assert nl == 1, nl                                                                       # the same place but another class is no partner
    try:
        G.compare_dumps(A["samples"], {"other": A["samples"]["t0"]})
        raise AssertionError("dumps of other samples must be refused")
    except ValueError:
        pass
    with tempfile.TemporaryDirectory() as tmp:
        pa, pb = os.path.join(tmp, "a.pkl"), os.path.join(tmp, "b.pkl")
        for path, obj in ((pa, A), (pb, dump([row(10.05, 0), row(20, 5), row(30, -5)], [0, 0, 8], [0.9, 0.8, 0.7], True))):
            with open(path, "wb") as f:
                pickle.dump(obj, f)
        assert G.tf32_gate(pa, pb) == 0 and G.tf32_gate(pa, pb, thr=0.01) == 1
        # overlay: GT 3 boxes in the sector (one at x 55 outside), CRN finds two of them
        gts = [{"group": "vehicle", "cx": 10.0, "cy": 0.0, "length": 4.5, "width": 2.0, "yaw": 0.0, "vx": 0.0, "vy": 0.0}, {"group": "pedestrian", "cx": 20.0, "cy": 3.0, "radius": 0.4, "vx": 0.0, "vy": 0.0},
               {"group": "vehicle", "cx": 30.0, "cy": 5.0, "length": 4.5, "width": 2.0, "yaw": 0.0, "vx": 0.0, "vy": 0.0}, {"group": "vehicle", "cx": 55.0, "cy": 0.0, "length": 4.5, "width": 2.0, "yaw": 0.0, "vx": 0.0, "vy": 0.0}]
        dets = [dict(gts[0], score=0.9, cls="car"), dict(gts[1], score=0.9, cls="pedestrian", length=0.8, width=0.8, yaw=0.0), dict(gts[2], score=0.1, cls="car")]       # the third is below the floor
        assert G.hits(dets, gts, 50.0, 0.3) == (3, 2), G.hits(dets, gts, 50.0, 0.3)
        rec = os.path.join(tmp, "rec.pkl")
        with open(rec, "wb") as f:
            pickle.dump({"det": {"scene-0001": {0.0: [(dets, gts)] * 3}, "scene-0002": {0.0: [(dets, gts)] * 2}}, "meta": {}}, f)
        png = os.path.join(tmp, "sub", "o.png")
        assert G.overlay(rec, png, n=4) == 0 and os.path.getsize(png) > 5000
        assert G.overlay(rec, png, n=20) == 0                                                # fewer frames than asked: all of them
    # the devkit sample filter: only the listed samples, an unknown sample refused, the real load_gt restored
    import nuscenes.eval.detection.evaluate as ev
    from nuscenes.eval.common.data_classes import EvalBoxes

    def fake(nusc, split, box_cls, verbose=False):
        g = EvalBoxes()
        for t in "abc":
            g.boxes[t] = [t]
        return g
    with mock.patch.object(ev, "load_gt", fake):
        with D.gt_of_samples({"a", "b"}):
            assert sorted(ev.load_gt(None, "val", None).sample_tokens) == ["a", "b"]
        assert ev.load_gt is fake
        try:
            with D.gt_of_samples({"a", "zz"}):
                ev.load_gt(None, "val", None)
            raise AssertionError("a sample without GT must be refused")
        except SystemExit:
            pass
        assert ev.load_gt is fake


def main():
    check_known_answer()
    check_against_devkit()
    n = check_mutants()
    check_chain()
    check_gates()
    print(f"CRN export test ok: planted box (centre, yaw, velocity, size, ten classes) by hand and against the devkit's Box, {n} planted bugs all caught, the chain on a toy store "
          "(a perfect CRN = the GT, mAP2 1.0), TF32 / overlay / devkit-filter gates")


if __name__ == "__main__":
    main()
