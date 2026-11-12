import gzip, pickle, os, glob, sys
store = "/mnt/c/Auto-research/data/autogt/lane/0702a065c6"
f = sorted(glob.glob(store + "/*.pkl.gz"))[300]
g = pickle.load(gzip.open(f, "rb"))
print("n pieces", len(g["pieces"]))
for p in g["pieces"][:8]:
    print(p["id"], p["piece"], p["part"], p["kind"], "len %.1f" % p["length"], "succ", p["succ"], "pred", p["pred"], "tokens", p["tokens"][:2], "segments", len(p["segments"]), "road", p["road"])
print("segments example", g["pieces"][0]["segments"][:2])
d = pickle.load(open("/mnt/c/Auto-research/bev_fusion_lane_center/temp/e276/predfull_e273_p_s0.pkl", "rb"))[300]
print("map_ids sample", d["map_ids"][:6], len(d["map_ids"]), "assign_P", d["assign_P"].shape)
