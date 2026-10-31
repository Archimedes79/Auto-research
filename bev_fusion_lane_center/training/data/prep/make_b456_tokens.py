"""D7 (2026-10-25): token lists and split assignment of the 255 new scenes (nuScenes parts 4-6), written to new files only.

    wsl bash tools/env/_py14.sh make_b456_tokens.py      (training/data/prep/, moved from temp/loop19/ on 2026-10-31, D24)

New scenes = scenes with camera AND radar keyframes on disk that are not in data/splits.json. Official train scenes -> part 'train', official val scenes -> part 'val'
(the parts of the feature arrays, as in b123). Order: by scene name, then by time inside the scene.
Split assignment (D7 option b): train2_new = official train without singapore-hollandvillage; unseen = the singapore-hollandvillage scenes (a location not in
b123); confirm = official val (never used for tuning; confirms a champion candidate before its TEST check). TRAIN / VAL / TEST of b123 stay as they are.
Writes: ~/exp_gcar_b456/feats/{train,val}_tokens.json and data/splits_b456.json (indices into those token lists).
"""
import collections
import json
import os

from nuscenes.utils import splits as S

HOME = os.path.expanduser("~")
ROOT = f"{HOME}/data/nuscenes_full"
META = f"{ROOT}/v1.0-trainval"
BFLC = os.environ["BFLC_ROOT"]
OUT = f"{HOME}/exp_gcar_b456/feats"
UNSEEN = "singapore-hollandvillage"

scene = {s["token"]: s for s in json.load(open(f"{META}/scene.json"))}
loc = {l["token"]: l["location"] for l in json.load(open(f"{META}/log.json"))}
samples = json.load(open(f"{META}/sample.json"))
on_disk = collections.defaultdict(set)
for x in json.load(open(f"{META}/sample_data.json")):
    f = x["filename"]
    if x["is_key_frame"] and (f.startswith("samples/CAM_FRONT/") or f.startswith("samples/RADAR_FRONT/")) and os.path.exists(f"{ROOT}/{f}"):
        on_disk[x["sample_token"]].add(f.split("/")[1])
used = json.load(open(f"{BFLC}/data/splits.json"))["scenes"]
used = set(used["train"]) | set(used["val"]) | set(used["test"])
by_scene = collections.defaultdict(list)
for s in samples:
    name = scene[s["scene_token"]]["name"]
    if name not in used and on_disk[s["token"]] == {"CAM_FRONT", "RADAR_FRONT"}:
        by_scene[name].append((s["timestamp"], s["token"]))
scene_loc = {scene[t]["name"]: loc[scene[t]["log_token"]] for t in scene}
parts = {"train": [n for n in sorted(by_scene) if n in set(S.train)], "val": [n for n in sorted(by_scene) if n in set(S.val)]}
assert not set(by_scene) - set(parts["train"]) - set(parts["val"]), "a new scene is in neither official split"
tokens, rows = {}, {}
for part, names in parts.items():
    tokens[part] = [t for n in names for _, t in sorted(by_scene[n])]
    rows[part] = {n: [i for i, t in enumerate(tokens[part]) if t in {tk for _, tk in by_scene[n]}] for n in names}
os.makedirs(OUT, exist_ok=True)
for part in ("train", "val"):
    json.dump(tokens[part], open(f"{OUT}/{part}_tokens.json", "w"))
train2 = [n for n in parts["train"] if scene_loc[n] != UNSEEN]
unseen = [n for n in parts["train"] if scene_loc[n] == UNSEEN]
idx = lambda part, names: [i for n in names for i in rows[part][n]]  # noqa: E731
out = dict(
    note="b456 = nuScenes parts 4-6 (D1, D7 option b, 2026-10-25). Indices into ~/exp_gcar_b456/feats/{train,val}_tokens.json. TRAIN2 = b123 TRAIN + train2_new; "
         "unseen = singapore-hollandvillage (location not in b123); confirm = official val scenes, never used for tuning. b123 TRAIN / VAL / TEST unchanged (data/splits.json).",
    train2_new=idx("train", train2), unseen=idx("train", unseen), confirm=idx("val", parts["val"]),
    scenes=dict(train2_new=train2, unseen=unseen, confirm=parts["val"]),
    locations={k: dict(collections.Counter(scene_loc[n] for n in v)) for k, v in (("train2_new", train2), ("unseen", unseen), ("confirm", parts["val"]))},
    n=dict(train_tokens=len(tokens["train"]), val_tokens=len(tokens["val"]), train2_new=len(idx("train", train2)), unseen=len(idx("train", unseen)), confirm=len(idx("val", parts["val"]))))
json.dump(out, open(f"{BFLC}/data/splits_b456.json", "w"), indent=1)
print(out["n"], {k: len(v) for k, v in out["scenes"].items()}, out["locations"])
