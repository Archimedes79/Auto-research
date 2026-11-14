"""crn_devkit_gate - the nuScenes devkit evaluation of CRN's own json on the SAMPLES OF THE INFO FILE only (card t26, G0, export gate (c): a sanity gate of the pipeline, not a reproduction).

Runs in the env `crn` (the devkit and CRN's evaluator are there), CPU. The devkit loads the whole trainval tables (about 8 GB of RAM, the figure of GtGenerator/light_nusc.py; not measured here, a few
minutes); `--light` builds minimal tables of the info samples instead (peak resident 1.9 GB and 1 minute, measured on the oracle json):
    ~/miniconda3/envs/crn/bin/python evaluation/crn/crn_devkit_gate.py RESULTS_JSON [--info INFOS.pkl] [--target 0.473] [--tol 0.05] [--partial] [--light]

CRN's own evaluator (evaluators/det_evaluators.py DetNuscEvaluator._evaluate_single -> the devkit's NuScenesEval, split `val`) asserts that the predictions cover every sample of the 150
val scenes ("Samples in split doesn't match samples in predictions"); we hold images and radar for 1706 samples of 43 of them. The subclass below leaves CRN's code and the devkit
untouched and, for the length of the call, hands `load_gt` a version that keeps only the samples of the list (a sample of the list that has no GT is an error). The list = the sample
tokens of the info file (the 1706 that crn_export.py ran); with --partial the tokens of the json (a smoke run).
Prints mAP, NDS and the AP of every class (mean over the four distances) and the verdict; writes `devkit_gate.json` next to the results json. Target: the repository's README, R50
mAP 47.3 (NDS 56.2) on the full val set; +-0.05 is the spread one expects of a 43-scene subset. NOTE on what it says: a pass means the boxes, the classes and the global frame of the
export are right (a wrong frame or class order would give mAP near 0); it does not say the network was reproduced.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import pickle
import shutil
import sys

CRN = os.path.expanduser("~/work/CRN")


@contextlib.contextmanager
def gt_of_samples(tokens):
    """for the length of the block the devkit's `load_gt` returns only the GT of `tokens`"""
    import nuscenes.eval.detection.evaluate as ev
    real, keep = ev.load_gt, set(tokens)

    def only(nusc, eval_split, box_cls, verbose=False):
        gt = real(nusc, eval_split, box_cls, verbose=verbose)
        unknown = keep - set(gt.sample_tokens)
        if unknown:
            raise SystemExit(f"{len(unknown)} samples of the list are not samples of the '{eval_split}' split (first {sorted(unknown)[0]})")
        for t in set(gt.sample_tokens) - keep:
            del gt.boxes[t]
        return gt
    ev.load_gt = only
    try:
        yield
    finally:
        ev.load_gt = real


def subset_evaluator(tokens, class_names, data_root, output_dir):
    """CRN's DetNuscEvaluator whose devkit run scores only `tokens`"""
    from evaluators.det_evaluators import DetNuscEvaluator

    class SubsetDetNuscEvaluator(DetNuscEvaluator):
        def _evaluate_single(self, result_path, *args, **kwargs):
            with gt_of_samples(tokens):
                return super()._evaluate_single(result_path, *args, **kwargs)
    return SubsetDetNuscEvaluator(class_names=class_names, data_root=data_root, output_dir=output_dir)


SMALL_TABLES = ("attribute", "calibrated_sensor", "category", "instance", "log", "map", "scene", "sensor", "visibility")


def light_tables(infos, data_root, out):
    """--light: the nuScenes tables of the info samples only, in `out` (the devkit then needs about 2 GB instead of the trainval tables' ~8 GB): sample.json of the listed samples, their LIDAR_TOP
    key frames (sample_data) and ego poses, their annotations (all taken from the info file, which gen_info.py made from the same tables), and the small tables copied whole. Idempotent."""
    table_dir = os.path.join(out, "v1.0-trainval")
    if os.path.exists(os.path.join(table_dir, "sample_annotation.json")):
        return out
    os.makedirs(table_dir, exist_ok=True)
    real = os.path.join(data_root, "v1.0-trainval")
    for t in SMALL_TABLES:
        shutil.copyfile(os.path.join(real, t + ".json"), os.path.join(table_dir, t + ".json"))
    with open(os.path.join(real, "sample.json")) as f:
        by_token = {s["token"]: s for s in json.load(f)}
    keys = ("token", "sample_token", "instance_token", "visibility_token", "attribute_tokens", "translation", "size", "rotation", "prev", "next", "num_lidar_pts", "num_radar_pts")
    samples, data, ego, ann = [], [], {}, []
    for i in infos:
        lid, e = i["lidar_infos"]["LIDAR_TOP"], i["lidar_infos"]["LIDAR_TOP"]["ego_pose"]
        samples.append(by_token[i["sample_token"]])
        ego[e["token"]] = {k: e[k] for k in ("token", "timestamp", "rotation", "translation")}
        data.append({"token": "sd_" + i["sample_token"], "sample_token": i["sample_token"], "ego_pose_token": e["token"], "calibrated_sensor_token": lid["calibrated_sensor"]["token"],
                     "timestamp": lid["timestamp"], "fileformat": "pcd", "is_key_frame": True, "height": 0, "width": 0, "filename": lid["filename"], "prev": "", "next": ""})
        ann += [{k: a[k] for k in keys} for a in i["ann_infos"]]
    for name, rows in (("sample", samples), ("sample_data", data), ("ego_pose", list(ego.values())), ("sample_annotation", ann)):
        with open(os.path.join(table_dir, name + ".json"), "w") as f:
            json.dump(rows, f)
    os.symlink(os.path.join(data_root, "maps"), os.path.join(out, "maps"))              # the devkit opens the map masks at start
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("results", help="results_nusc.json of crn_export.py")
    p.add_argument("--info", default=os.path.join(CRN, "data", "nuScenes", "nuscenes_infos_val.pkl"))
    p.add_argument("--data-root", default=os.path.join(CRN, "data", "nuScenes"))
    p.add_argument("--target", type=float, default=0.473, help="CRN R50 mAP of the README (47.3)")
    p.add_argument("--tol", type=float, default=0.05)
    p.add_argument("--partial", action="store_true", help="the samples of the json instead of the info file's (a smoke run)")
    p.add_argument("--light", action="store_true", help="minimal tables of the info samples (written next to the json, ~2 GB RAM) instead of the whole trainval tables (~8 GB)")
    a = p.parse_args(argv)
    sys.path.insert(0, CRN)
    results = os.path.abspath(a.results)
    with open(results) as f:
        have = set(json.load(f)["results"])
    with open(a.info, "rb") as f:
        infos = pickle.load(f)
    listed = {i["sample_token"] for i in infos}
    tokens = have if a.partial else listed
    if have != tokens:
        raise SystemExit(f"the json holds {len(have)} samples, the info file {len(listed)} ({len(have - listed)} in the json only, {len(listed - have)} in the info only); --partial scores the json's")
    from exps.base_exp import CLASSES
    out = os.path.dirname(results)
    data_root = light_tables(infos, a.data_root, os.path.join(out, "light_tables")) if a.light else a.data_root
    ev = subset_evaluator(tokens, CLASSES, data_root, out)
    print(f"devkit evaluation of {len(tokens)} samples ({'json' if a.partial else 'info file'} list), data root {data_root}", flush=True)
    ev._evaluate_single(results)                                                  # writes metrics_summary.json next to the results
    with open(os.path.join(out, "metrics_summary.json")) as f:
        m = json.load(f)
    ap = m["mean_dist_aps"]
    ok = abs(m["mean_ap"] - a.target) <= a.tol
    print(f"mAP {m['mean_ap']:.4f}  NDS {m['nd_score']:.4f}  | car AP {ap['car']:.3f}  pedestrian AP {ap['pedestrian']:.3f} | " + "  ".join(f"{k} {v:.3f}" for k, v in ap.items() if k not in ("car", "pedestrian")))
    print(f"gate (c): {'PASS' if ok else 'FAIL'} - mAP {m['mean_ap']:.4f} against the README's {a.target:.3f} +- {a.tol:g} on {len(tokens)} samples")
    with open(os.path.join(out, "devkit_gate.json"), "w") as f:
        json.dump({"samples": len(tokens), "mAP": m["mean_ap"], "NDS": m["nd_score"], "AP": ap, "target": a.target, "tol": a.tol, "pass": ok, "results": results}, f, indent=1)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
