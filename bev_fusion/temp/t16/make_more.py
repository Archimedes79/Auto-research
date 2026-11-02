"""scratch of the t16 follow-up (D46): the scene lists of the "more data" arm. Writes temp/t16/more_scenes.json ({new_train: [...272 scenes of nuScenes
parts 7 - 10 that are in the trainval train split and have a DualBEV shard]}) and, with --config, training/configs/t17_data.json = t16_full with these scenes added to
train_scenes (the 82 val scenes stay: the arm is read against t16.full_s0 on the same frames) and the feature caches of data/cache/all/.
Run from bev_fusion/: python make_more.py [--config]"""
import json
import os
import sys

from nuscenes.utils.splits import create_splits_scenes

sp = create_splits_scenes()
cfg_path = "training/configs/t16_full.json"
c = json.load(open(cfg_path, encoding="utf-8"))
have = set()
for d in ("cam_feats", "cam_feats_parts7_10"):
    p = "data/cache/" + d
    have |= {f[:-4] for f in os.listdir(p) if f.endswith(".npy")}
new = sorted((set(sp["train"]) & have) - set(c["train_scenes"]))
json.dump({"new_train": new}, open("temp/t16/more_scenes.json", "w"), indent=1)
print(len(new), "new train scenes")
if "--config" in sys.argv:
    t = dict(c)
    t["_about"] = ("t17 more data (D46, tomorrow-afternoon plan): t16_full with the 272 train scenes of nuScenes parts 7 - 10 added (700 train scenes, the 82 val scenes unchanged, "
                   "so it is read against t16.full_s0 on the same frames); everything else is t16_full's: the recipe, the blocks, 6000 steps. The BEVFusion shards of the new scenes "
                   "are built by temp/t16/prep_more.sh into data/cache/all/ (the DualBEV and BEVFusion shards of all scenes are linked there)")
    t["name"] = "t17_data"
    t["feature_cache"] = "data/cache/all/cam_feats"
    t["feature_cache2"] = "data/cache/all/cam_feats_bevfusion"
    t["train_scenes"] = list(c["train_scenes"]) + new
    json.dump(t, open("training/configs/t17_data.json", "w", encoding="utf-8"), indent=1)
    open("training/configs/t17_data.json", "a").write("\n")
    print("written training/configs/t17_data.json:", len(t["train_scenes"]), "train scenes,", len(t["val_scenes"]), "val scenes")
