import os, pickle, sys
R = os.environ["BFLC_ROOT"]
sys.path.insert(0, R + "/evaluation")
import jwire_data as JD
tr = pickle.load(open(JD.cache_path("train", 0, 200), "rb")); va = pickle.load(open(JD.cache_path("val", 0), "rb"))
for n, r in (("TRAIN 200", tr), ("VAL 873", va)):
    p = JD.pilot_stats(r)
    print(n, {k: round(v, 3) for k, v in p.items() if k not in ("counts",)})
