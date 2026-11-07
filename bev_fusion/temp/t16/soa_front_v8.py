"""soa_front_v8.py - the pre-trained DualBEV (front camera only) on the 1000 val frames of runs/t16_full_s0, scored against the v8 ground truth.

Writes a scene_records-format pickle that `evaluation/front_compare.py` (`bev_suite.py front`) takes as a `LABEL=FILE.pkl` system:
  {"seg": {}, "det": {scene: {0.0: [(dets, gt boxes) per frame]}}, "detr": {scene: {}}, "meta": {"frames": [tokens], "gt_signature": ...}}

  frames    runs/<ref>/eval/suite/quant/eval_curves_fusion_vel_n1000/frames.jsonl (scene / index / key), the 1000 frames of the table of record
  GT        data/cache/gt/<token>_<signature>.npz, the file the trainer / suite dataset reads (signature from runs/<ref>/gt.json: v8), read only;
            boxes by evaluation.metrics.gt_boxes_from_meta of the 'cur' GT, as bev_suite.py eval (bev_views.gt_boxes) keeps them
  detector  training/legacy/dualbev_front.py (our port, weights models/pretrained/dualbev-r50-cbgs.pth unchanged), CAM_FRONT only, single frame,
            3 x 3 peaks, k 100, score >= 0.05 (as temp/t4/dualbev_front_eval.py, the script of the older record runs/t4_dualbev_front)
  input     the image / camera vector exactly as FrontRadarDataset builds them (its `_image`, the same K / T_cam_car / camera_vector lines); the nuScenes
            tables are NOT loaded through the devkit (trainval needs several GB of RAM): the CAM_FRONT file name and calibration of the val keyframes are
            read once from the json tables by a streaming parser into temp/t16/soa/soa_front_v8_inputs.json

CPU only (CUDA_VISIBLE_DEVICES is emptied before torch is imported), 2 threads.  From the bev_fusion folder, under nice:
  python temp/t16/soa_front_v8.py --n 1000 --out temp/t16/soa/dualbev_front_v8_n1000.pkl          the record (resumable: <out>.partial)
  python temp/t16/soa_front_v8.py --n 5 --out temp/t16/soa/dualbev_front_v8_n5.pkl                a test on 5 frames spread over the 1000
  python temp/t16/soa_front_v8.py --gates                                                         sanity gates (a) and (c), no model
  python temp/t16/soa_front_v8.py --check-old runs/t4_dualbev_front/eval/scene_records.pkl        sanity gate (b): the old record, 5 frames recomputed
"""
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""                     # the GPU is never touched
os.environ["OMP_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"] = "2"

import argparse  # noqa: E402
import json  # noqa: E402
import pickle  # noqa: E402
import resource  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(2)

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
for _p in (os.path.join(ROOT, "training", "legacy"), os.path.join(ROOT, "training"), ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(ROOT)

REF = "runs/t16_full_s0"
OTHER = "runs/t17_data"
FRAMES = "eval/suite/quant/eval_curves_fusion_vel_n1000/frames.jsonl"
GT_DIR = "data/cache/gt"
TABLES = os.path.expanduser("~/data/nuscenes_full/v1.0-trainval")
DATAROOT = os.path.expanduser("~/data/nuscenes_full")
INPUTS = "temp/t16/soa/soa_front_v8_inputs.json"
CKPT = "models/pretrained/dualbev-r50-cbgs.pth"


# ------------------------------------------------------------------------------------------------ frames, GT
def ref_rows(run=REF):
    """[{'scene', 'index', 'key'}] of the table-of-record frames, in the order of the store (scene by scene, time order)"""
    out = []
    with open(os.path.join(run, FRAMES)) as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                out.append({"scene": d["scene"], "index": d["index"], "key": d["key"]})
    return out


def gt_signature(run=REF):
    with open(os.path.join(run, "gt.json")) as f:
        return json.load(f)["gt_signature"]


def read_gt(token, sig):
    """the dict of the dataset's GT cache file (keys cur / prev / next); read only, a missing file is an error (nothing is ever built here)"""
    path = os.path.join(GT_DIR, f"{token}_{sig}.npz")
    with np.load(path, allow_pickle=True) as z:
        return z["gt"].item()


def gt_boxes_of(g, groups):
    """box dicts of the 'cur' GT, as bev_views.gt_boxes(sample.reference[0.0], groups) of bev_suite.py eval"""
    from evaluation.metrics import gt_boxes_from_meta
    return [b for b in gt_boxes_from_meta(g["cur"]["meta"]["boxes"]) if b["group"] in groups]


def groups():
    from losses import NUSC_GROUPS
    return [g for g, _ in NUSC_GROUPS]


# ------------------------------------------------------------------------------------------------ inputs without the devkit
def iter_json_array(path, chunk=1 << 25):
    """the objects of a (pretty-printed) json array file, one at a time, without holding the file"""
    dec = json.JSONDecoder()
    with open(path, "r", encoding="utf-8") as f:
        buf = f.read(chunk)
        i = buf.index("[") + 1
        while True:
            n = len(buf)
            while i < n and buf[i] in " \t\r\n,":
                i += 1
            if i >= n:
                more = f.read(chunk)
                if not more:
                    return
                buf, i = buf[i:] + more, 0
                continue
            if buf[i] == "]":
                return
            try:
                obj, j = dec.raw_decode(buf, i)
            except json.JSONDecodeError:
                more = f.read(chunk)
                if not more:
                    raise
                buf, i = buf[i:] + more, 0
                continue
            yield obj
            i = j


def build_inputs(path=INPUTS):
    """CAM_FRONT file name + calibration of every val keyframe the trainer's dataset has (the scenes of the ref run's config, keyframes with a previous and a
    next one: FrontRadarDataset.samples), in the order of sample.json (= the devkit's nusc.sample); cached as json"""
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    t0 = time.time()
    with open(os.path.join(REF, "config.json")) as f:
        names = set(json.load(f)["val_scenes"])
    scene_name = {s["token"]: s["name"] for s in json.load(open(os.path.join(TABLES, "scene.json")))}
    order, scene_of = [], {}
    for s in json.load(open(os.path.join(TABLES, "sample.json"))):
        if scene_name[s["scene_token"]] in names and s["prev"] and s["next"]:
            order.append(s["token"])
            scene_of[s["token"]] = scene_name[s["scene_token"]]
    want = set(order)
    cs_table = {c["token"]: c for c in json.load(open(os.path.join(TABLES, "calibrated_sensor.json")))}
    frames, seen = {}, 0
    for sd in iter_json_array(os.path.join(TABLES, "sample_data.json")):
        seen += 1
        if sd["is_key_frame"] and sd["sample_token"] in want and sd["filename"].startswith("samples/CAM_FRONT/"):
            cs = cs_table[sd["calibrated_sensor_token"]]
            frames[sd["sample_token"]] = {"scene": scene_of[sd["sample_token"]], "sd": sd["token"], "filename": sd["filename"],
                                          "cs": {k: cs[k] for k in ("translation", "rotation", "camera_intrinsic")}}
            if len(frames) == len(want):
                break
    missing = want - set(frames)
    if missing:
        raise SystemExit(f"{len(missing)} val keyframes without a CAM_FRONT sample_data")
    out = {"scenes": sorted(names), "order": order, "frames": frames}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w") as f:
        json.dump(out, f)
    os.replace(path + ".tmp", path)
    print(f"inputs: {len(order)} val keyframes of {len(names)} scenes, {seen} sample_data records read, {time.time() - t0:.0f} s -> {path}", flush=True)
    return out


class Inputs:
    """the network input of one keyframe, built by the dataset's own code where it exists (FrontRadarDataset._image) and the same lines as `_cam_frames`"""

    def __init__(self, info):
        from camera_encoder import nuscenes_front_ida
        from nuscenes_dataset import DST_HW, FrontRadarDataset
        self.info = info
        self.ds = FrontRadarDataset.__new__(FrontRadarDataset)      # no devkit tables: only what _image reads (root, ida)
        self.ds.root, self.ida = DATAROOT, nuscenes_front_ida(dst_hw=DST_HW)
        self.ds.ida = self.ida

    def __call__(self, token):
        from camera_encoder import camera_vector
        from nuscenes_dataset import pose_matrix
        fr = self.info["frames"][token]
        cs = fr["cs"]
        K = torch.tensor(cs["camera_intrinsic"], dtype=torch.float32)
        T_c2e = torch.tensor(pose_matrix(cs), dtype=torch.float32)
        cv = camera_vector(K[None], self.ida[None, :2, :2], self.ida[None, :2, 2], T_c2e[None])[0]
        return {"img": self.ds._image({"filename": fr["filename"]}), "K": K, "T_cam_car": torch.linalg.inv(T_c2e), "cam_vec": cv}


# ------------------------------------------------------------------------------------------------ the detector
def load_model():
    from dualbev_front import DualBEVFront
    assert not torch.cuda.is_available(), "this script runs on the CPU only"
    model = DualBEVFront().eval()
    rep = model.load_pretrained(CKPT)
    assert rep["missing"] == [] and rep["unexpected"] == ["img_view_transformer.prob.ce_loss.pos_weight"], rep
    return model


@torch.no_grad()
def detect(model, inp, ida):
    """dets of one frame, as temp/t4/dualbev_front_eval.py: RGB input, decode(k=100, threshold=0.05); -> (dets, model ms)"""
    from dualbev_front import decode
    t = time.time()
    outs = model(inp["img"][None], inp["cam_vec"][None], inp["K"][None], inp["T_cam_car"][None], ida)
    outs = [{k: v.float().cpu() for k, v in o.items()} for o in outs]
    ms = (time.time() - t) * 1e3
    return decode(outs, k=100, threshold=0.05)[0], ms


def save_atomic(obj, path):
    with open(path + ".tmp", "wb") as f:
        pickle.dump(obj, f)
    os.replace(path + ".tmp", path)


# ------------------------------------------------------------------------------------------------ modes
def run(a):
    rows = ref_rows()
    assert len(rows) == 1000, len(rows)
    if a.n < len(rows):
        rows = [rows[i] for i in sorted({int(i) for i in np.linspace(0, len(rows) - 1, a.n).round()})]
    sig = gt_signature()
    missing = [r["key"] for r in rows if not os.path.exists(os.path.join(GT_DIR, f"{r['key']}_{sig}.npz"))]
    if missing:
        raise SystemExit(f"{len(missing)} GT cache files of signature {sig} are missing (first {missing[0]}); nothing is built here")
    info = build_inputs()
    bad = [r["key"] for r in rows if r["key"] not in info["frames"] or info["frames"][r["key"]]["scene"] != r["scene"]]
    if bad:
        raise SystemExit(f"{len(bad)} frames of the list are not val keyframes of their scene in the inputs table (first {bad[0]})")
    grp, inputs, model = groups(), Inputs(info), load_model()
    part = a.out + ".partial"
    done = pickle.load(open(part, "rb")) if os.path.exists(part) else {}
    t0, secs, mss, new = time.time(), [], [], 0
    for k, r in enumerate(rows):
        tok = r["key"]
        if tok in done:
            continue
        t1 = time.time()
        dets, ms = detect(model, inputs(tok), inputs.ida)
        done[tok] = (r["scene"], dets, gt_boxes_of(read_gt(tok, sig), grp))
        secs.append(time.time() - t1)
        mss.append(ms)
        new += 1
        if new % 5 == 0 or k == len(rows) - 1:
            print(f"{len(done)}/{len(rows)} frames, {time.time() - t0:.0f} s, {np.mean(secs):.2f} s/frame (model {np.mean(mss) / 1e3:.2f} s), "
                  f"max RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.2f} GB", flush=True)
        if new % 25 == 0:
            save_atomic(done, part)
    det, order = {}, []
    for r in rows:
        det.setdefault(r["scene"], {0.0: []})[0.0].append(done[r["key"]][1:])
        order.append(r["key"])
    meta = {"frames": order, "gt_signature": sig, "producer": "temp/t16/soa_front_v8.py", "device": "cpu", "threads": 2,
            "model": "training/legacy/dualbev_front.py + models/pretrained/dualbev-r50-cbgs.pth (EMA, unchanged), CAM_FRONT only, single frame",
            "decode": {"k": 100, "threshold": 0.05}, "frames_from": os.path.join(REF, FRAMES), "gt_dir": GT_DIR,
            "s_per_frame_this_process": float(np.mean(secs)) if secs else None}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    save_atomic({"seg": {}, "det": det, "detr": {s: {} for s in det}, "meta": meta}, a.out)
    if os.path.exists(part):
        os.remove(part)
    print(f"done: {len(rows)} frames of {len(det)} scenes -> {a.out}; {np.mean(secs) if secs else float('nan'):.2f} s/frame in this process "
          f"({new} frames computed)", flush=True)
    return 0


def check_old(a):
    """gate (b): recompute a few frames of the older record (runs/t4_dualbev_front, GT older than v5) and compare the detections"""
    info = build_inputs()
    order = info["order"]
    sel = sorted({int(i) for i in np.linspace(0, len(order) - 1, 1000).round()})           # val_subset(ds, 1000) of the old script
    old_toks = [order[i] for i in sel]
    rows = ref_rows()
    print(f"old frame list reconstructed: {len(old_toks)} frames of {len(set(info['frames'][t]['scene'] for t in old_toks))} scenes; "
          f"{len(set(old_toks) & {r['key'] for r in rows})} of them are in the {len(rows)} frames of {REF}; "
          f"{len(order)} val keyframes in all", flush=True)
    with open(a.check_old, "rb") as f:
        old = pickle.load(f)["det"]
    pos, per_scene = {}, {}
    for t in old_toks:
        s = info["frames"][t]["scene"]
        pos[t] = per_scene.get(s, 0)
        per_scene[s] = pos[t] + 1
    assert sorted(per_scene) == sorted(old) and all(per_scene[s] == len(old[s][0.0]) for s in per_scene), "old record: other scenes / frames per scene"
    print("old record: same scenes and frames per scene as the reconstructed list", flush=True)
    pick = [old_toks[i] for i in sorted({int(i) for i in np.linspace(0, len(old_toks) - 1, a.check_n).round()})]
    model, inputs = load_model(), None
    inputs = Inputs(info)
    worst, rows_out = 0.0, []
    for tok in pick:
        dets, ms = detect(model, inputs(tok), inputs.ida)
        ods = old[info["frames"][tok]["scene"]][0.0][pos[tok]][0]
        same_n = len(dets) == len(ods)
        d = 0.0
        for x, y in zip(dets, ods):
            assert x["group"] == y["group"] and x.get("cls") == y.get("cls"), (x, y)
            d = max(d, max(abs(x[k] - y[k]) for k in ("score", "cx", "cy", "vx", "vy", "length", "width", "yaw")))
        worst = max(worst, d)
        rows_out.append((tok, info["frames"][tok]["scene"], pos[tok], len(dets), len(ods), d, ms / 1e3))
        print(f"  {tok[:8]} {info['frames'][tok]['scene']}[{pos[tok]}]: {len(dets)} dets now, {len(ods)} in the old record, max |diff| over "
              f"score/cx/cy/vx/vy/size/yaw {d:.2e}, {ms / 1e3:.1f} s", flush=True)
    ok = all(r[3] == r[4] for r in rows_out) and worst < 1e-3
    print(f"gate (b): {'PASS' if ok else 'FAIL'} - {len(pick)} frames, same number of detections in all: {all(r[3] == r[4] for r in rows_out)}, worst difference {worst:.2e}")
    return 0 if ok else 1


def gates(a):
    """gate (a) and gate (c), no model: the GT read from the cache against the GT inside the stored records of the runs, and the front-sector numbers of the
    runs by the front_compare path against the whole-area numbers of the table of record"""
    import evaluation.front_compare as FC
    sig, grp, rows = gt_signature(), groups(), ref_rows()
    print(f"frames: {len(rows)} of {len(set(r['scene'] for r in rows))} scenes; GT signature {sig} (gt.json of {REF}: {gt_signature(REF)}, of {OTHER}: {gt_signature(OTHER)})")
    recs = {}
    for run in (REF, OTHER):
        fr, meta = FC.load_records(os.path.join(run, "eval", "scene_records_n1000.pkl"))
        assert meta["frames"] == [r["key"] for r in rows] and meta["gt_signature"] == sig, f"{run}: other frames / GT signature in the pickle"
        recs[run] = fr
    n_box, bad = 0, {REF: 0, OTHER: 0}
    pos = {}
    for r in rows:
        j = pos.get(r["scene"], 0)
        pos[r["scene"]] = j + 1
        gb = gt_boxes_of(read_gt(r["key"], sig), grp)
        n_box += len(gb)
        for run in recs:
            bad[run] += int(gb != recs[run][r["scene"]][j][1])
    print(f"gate (c): GT boxes of the cache read by this script (the dataset's file) vs the GT boxes stored in the records of each run, {len(rows)} frames, "
          f"{n_box} boxes in all: frames that differ {bad} -> {'PASS' if not any(bad.values()) else 'FAIL'}")
    with open(os.path.join(REF, "eval", "by_n", "metrics_n1000.json")) as f:
        m = json.load(f)["metrics"]
    for run in (REF, OTHER):
        r60 = FC.compare([(run, recs[run])], x_max=60.0, n_boot=2)["systems"][0]
        r50 = FC.compare([(run, recs[run])], x_max=50.0, n_boot=2)["systems"][0]
        if run == REF:
            print("gate (a): the front_compare path at x_max 60 m (= the area of the table of record) against det/dt+0.00/<class>/all/ap@<t> of metrics_n1000.json")
        for c in FC.CLASSES:
            for t in (0.5, 1.0, 2.0, 4.0):
                key = f"det/dt+0.00/{c}/all/ap@{t:g}"
                if run == REF:
                    print(f"   {key:42s} table {m[key]:.6f}  front_compare(60 m) {r60['ap'][c][t]:.6f}  diff {r60['ap'][c][t] - m[key]:+.1e}")
        print(f"   {run}: front_compare mAP2 60 m {r60['map2']:.4f}, 50 m sector {r50['map2']:.4f}; vehicle AP@2 {r60['ap']['vehicle'][2.0]:.4f} (60 m) / "
              f"{r50['ap']['vehicle'][2.0]:.4f} (50 m); pedestrian AP@2 {r60['ap']['pedestrian'][2.0]:.4f} (60 m) / {r50['ap']['pedestrian'][2.0]:.4f} (50 m)")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n", type=int, default=1000, help="frames, spread evenly over the 1000 (default all)")
    ap.add_argument("--out", default="temp/t16/soa/dualbev_front_v8_n1000.pkl")
    ap.add_argument("--gates", action="store_true", help="gates (a) and (c), no model")
    ap.add_argument("--check-old", default=None, help="gate (b): the older DualBEV record (runs/t4_dualbev_front/eval/scene_records.pkl)")
    ap.add_argument("--check-n", type=int, default=5)
    a = ap.parse_args()
    if a.gates:
        return gates(a)
    if a.check_old:
        return check_old(a)
    return run(a)


if __name__ == "__main__":
    raise SystemExit(main())
