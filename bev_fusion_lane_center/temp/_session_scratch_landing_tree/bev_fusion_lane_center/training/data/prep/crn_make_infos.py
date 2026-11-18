"""CRN-Infos (nuscenes_infos_{train,val}.pkl) fuer genau die Szenen erzeugen, aus denen unsere Samples stammen.

Nutzt gen_info.generate_info aus dem CRN-Repo (liest nur Metadaten, keine Lidar-Punktwolken). Die Szenen werden aus den
bisherigen tokens.json (Teile 1-3: ~/exp_b123/feats) bzw. aus den Mini-Splits bestimmt, damit dieselben Samples und dieselbe
Train/Val-Aufteilung wie bei den Lidar-Experimenten verwendet werden.

Aufruf (Umgebung crn):
    python crn_make_infos.py --root ~/data/nuscenes_full --version v1.0-trainval --tokens ~/exp_b123/feats
    python crn_make_infos.py --root ~/data/nuscenes --version v1.0-mini --mini
"""
import argparse
import json
import os
import sys

import mmcv
from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits

sys.path.insert(0, os.path.expanduser("~/work/CRN/scripts"))
import gen_info  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--version", required=True)
ap.add_argument("--tokens", default="", help="Ordner mit train_tokens.json / val_tokens.json")
ap.add_argument("--mini", action="store_true", help="Mini: Szenen aus mini_train / mini_val")
args = ap.parse_args()
root = os.path.expanduser(args.root)
nusc = NuScenes(version=args.version, dataroot=root, verbose=False)
scene_of = {s["token"]: nusc.get("scene", s["scene_token"])["name"] for s in nusc.sample}

for split in ("train", "val"):
    if args.mini:
        names = set(getattr(splits, f"mini_{split}"))
    else:
        toks = json.load(open(os.path.join(os.path.expanduser(args.tokens), f"{split}_tokens.json")))
        names = {scene_of[t] for t in toks}
    infos = gen_info.generate_info(nusc, names)
    out = os.path.join(root, f"nuscenes_infos_{split}_crn.pkl")
    mmcv.dump(infos, out)
    print(f"{split}: {len(names)} Szenen, {len(infos)} Samples -> {out}")
