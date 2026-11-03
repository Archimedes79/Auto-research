"""throw-away: which of the GaussianCaR frames have a v8 GT cache file / radar cache file? numpy + json only, no devkit."""
import glob
import json
import os
import sys

root = os.path.expanduser("~/data/nuscenes_full/v1.0-trainval")
cfg = json.load(open("runs/t16_full_s0/config.json"))
sc = {s["token"]: s["name"] for s in json.load(open(f"{root}/scene.json"))}
val = set(cfg["val_scenes"])
samples = [s for s in json.load(open(f"{root}/sample.json")) if sc[s["scene_token"]] in val]
usable = {s["token"] for s in samples if s["prev"] and s["next"]}
print("val scenes", len(val), "val samples", len(samples), "with prev+next", len(usable))
tok = json.load(open(os.path.expanduser("~/exp_gcar_b123/feats/val_tokens.json")))
old_idx = json.load(open("temp/soa_gap/gcar_val_idx_in_our_val.json"))
print("gcar tokens", len(tok), "old rows", len(old_idx), "now usable", sum(t in usable for t in tok))
old_rows = [(i, tok[i]) for i in old_idx]
base_old = old_rows[:: max(1, len(old_rows) // 400)][:400]
print("old 400:", len(base_old), "still usable:", sum(t in usable for _, t in base_old))
SIG = "0.5_4_6a7aa7864a"
d = cfg["gt_cache"]
have = lambda t: os.path.exists(f"{d}/{t}_{SIG}.npz")
print("v8 GT cache: val samples with prev+next that have a file:", sum(have(t) for t in usable), "of", len(usable))
print("old 400 usable with GT file:", sum(have(t) for _, t in base_old if t in usable))
print("all gcar usable with GT file:", sum(have(t) for t in tok if t in usable), "of", sum(t in usable for t in tok))
# which other signatures exist for a missing token
miss = [t for _, t in base_old if t in usable and not have(t)]
print("missing in the old 400:", len(miss), miss[:2])
if miss:
    print(glob.glob(f"{d}/{miss[0]}_*"))
ok = [t for _, t in base_old if t in usable and have(t)]
if ok:
    print(glob.glob(f"{d}/{ok[0]}_*"))
rc = cfg["radar_cache"] or "data/cache/radar"
print("radar cache for usable gcar frames (glob token_*):", sum(bool(glob.glob(f"{rc}/{t}_*")) for _, t in base_old if t in usable), "of", sum(t in usable for _, t in base_old))
