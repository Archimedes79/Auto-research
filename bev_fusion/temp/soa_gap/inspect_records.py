import pickle
import sys

d = pickle.load(open(sys.argv[1], "rb"))
print(d.keys())
det = d["det"]
sc = next(iter(det))
print("scenes", len(det), "example", sc, "dts", list(det[sc].keys()))
fr = det[sc][0.0]
print("frames in scene", len(fr))
dets, gts = fr[0]
print("n dets", len(dets), "n gt", len(gts))
print("det keys", sorted(dets[0].keys()) if dets else None)
print("det example", {k: (round(v, 2) if isinstance(v, float) else v) for k, v in dets[0].items()} if dets else None)
print("gt keys", sorted(gts[0].keys()) if gts else None)
print("n frames total dt0", sum(len(v[0.0]) for v in det.values()))
