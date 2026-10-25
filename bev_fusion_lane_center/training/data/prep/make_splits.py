"""Splits fuer die Graph-Extraktion (E27 ff.): scene-disjunkt, nach den offiziellen nuScenes-Szenen.
TRAIN = offizielle Train-Szenen der Teile 1-3 (212 Szenen, 8.414 Samples); VAL und TEST = offizielle Val-Szenen (43 Szenen), abwechselnd je Ort auf VAL (Abstimmung, Modellwahl) und TEST (nur Endwerte) verteilt.
Zusaetzlich TRAIN_EVAL = jedes 8. Trainings-Sample (Obergrenzen-/Ueberanpassungs-Diagnose). Ausgabe: ~/exp_gcar_b123/splits.json (Indizes in die Token-Listen train_tokens.json / val_tokens.json) und loop2/splits.json."""
import json
import os

home = os.path.expanduser("~")
exp = f"{home}/exp_crn_b123"
meta = f"{home}/data/nuscenes_full/v1.0-trainval"
sample_scene = {s["token"]: s["scene_token"] for s in json.load(open(f"{meta}/sample.json"))}
scene = {s["token"]: s for s in json.load(open(f"{meta}/scene.json"))}
log_loc = {l["token"]: l["location"] for l in json.load(open(f"{meta}/log.json"))}
tr = json.load(open(f"{exp}/feats/train_tokens.json"))
va = json.load(open(f"{exp}/feats/val_tokens.json"))
info = lambda t: (scene[sample_scene[t]]["name"], log_loc[scene[sample_scene[t]]["log_token"]])
vs = sorted({info(t) for t in va}, key=lambda x: (x[1], x[0]))
val_scenes, test_scenes = [], []
by_loc = {}
for n, l in vs:
    by_loc.setdefault(l, []).append(n)
for l, names in by_loc.items():
    for k, n in enumerate(names):
        (val_scenes if k % 2 == 0 else test_scenes).append(n)
val_idx = [i for i, t in enumerate(va) if info(t)[0] in set(val_scenes)]
test_idx = [i for i, t in enumerate(va) if info(t)[0] in set(test_scenes)]
tr_scenes = sorted({info(t)[0] for t in tr})
out = dict(
    train=list(range(len(tr))), train_eval=list(range(0, len(tr), 8)), val=val_idx, test=test_idx,
    scenes=dict(train=tr_scenes, val=sorted(val_scenes), test=sorted(test_scenes)),
    locations={l: len(v) for l, v in by_loc.items()},
    n=dict(train=len(tr), train_eval=len(range(0, len(tr), 8)), val=len(val_idx), test=len(test_idx)),
    note="Indizes: train/train_eval in train_tokens.json, val/test in val_tokens.json (offizielle Val-Szenen, scene-disjunkt)")
os.makedirs(f"{home}/exp_gcar_b123", exist_ok=True)
json.dump(out, open(f"{home}/exp_gcar_b123/splits.json", "w"), indent=1)
json.dump(out, open("/mnt/c/ML-Experiment/bev_fusion_lane_center/data/splits.json", "w"), indent=1)
print(out["n"], "Szenen:", {k: len(v) for k, v in out["scenes"].items()}, "Orte:", out["locations"])
