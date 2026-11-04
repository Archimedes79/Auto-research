"""throw-away: gate 1 in aggregate. Our CPU counts of the frames in soa_maps_v8 OUT.json against the stored suite counts (GPU run) of the same frames:
the headline numbers from the summed counts of both, and the share of keys that are equal. usage: soa_maps_v8_gate.py OUT.json  (run from bev_fusion)"""
import json
import sys

import numpy as np

out = sys.argv[1]
d = json.load(open(out))
z = np.load(out[:-5] + "_frames.npz")
store_dir = "runs/t16_full_s0/eval/suite/quant/" + d["stored_check"]["store"]
frames = {}
for line in open(store_dir + "/frames.jsonl"):
    fr = json.loads(line)
    frames[fr["key"]] = fr
checked = [i for i, f in enumerate(d["frames"]) if f.get("checked")]
toks = [d["frames"][i]["token"] for i in checked]
cache = {}


def stored(tok, key):
    fr = frames[tok]
    if fr["chunk"] not in cache:
        cache[fr["chunk"]] = np.load(f"{store_dir}/counts_seg_{fr['chunk']:05d}.npz")
    zz = cache[fr["chunk"]]
    keys = list(map(str, zz[f"{fr['pos']}|keys"]))
    j = keys.index(key)
    return zz[f"{fr['pos']}|cnt"][j], zz[f"{fr['pos']}|n_gt"][j]


def iou(c, t=0, zc=False):
    p_hit, p_miss, g_hit, g_miss = c[t]
    if zc:
        p, r = p_hit / (p_hit + p_miss), g_hit / (g_hit + g_miss)
        return 2 * p * r / (p + r)
    tp = (p_hit + g_hit) / 2
    return tp / (tp + p_miss + g_miss)


rows = [("drivable_area", 0, False), ("drivable_area", 1, False), ("ped_crossing", 0, False), ("walkway", 0, False), ("carpark_area", 0, False),
        ("lane_divider", 1, False), ("lane_divider", 2, False), ("road_divider", 2, False), ("lane_divider:zc", 1, True), ("road_divider:zc", 1, True)]
print(f"{len(checked)} frames in the stored evaluation {d['stored_check']['store']}; summed counts, mine (CPU) / stored (GPU) / difference")
nkeys = neq = 0
for layer, t, zc in rows:
    for b in ("all", "0-20", "20-40", "40-60"):
        mine = z[f"ours|0.5|+0.00|{layer}|all|{b}|cnt"][checked].sum(0)
        theirs = sum(stored(tk, f"fov|+0.00|{layer}|all|{b}")[0] for tk in toks)
        if b == "all":
            print(f"  {layer:16s} tol idx {t} {'f1' if zc else 'iou'}  {iou(mine, t, zc):.4f} / {iou(theirs, t, zc):.4f} / {iou(mine, t, zc) - iou(theirs, t, zc):+.5f}")
        for k, tk in enumerate(toks):
            nkeys += 1
            neq += int(np.array_equal(z[f"ours|0.5|+0.00|{layer}|all|{b}|cnt"][checked[k]], stored(tk, f"fov|+0.00|{layer}|all|{b}")[0]))
print(f"per-frame count arrays equal: {neq} of {nkeys} ({100 * neq / nkeys:.1f} %)")
