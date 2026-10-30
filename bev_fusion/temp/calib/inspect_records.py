"""One-off (t4_soa_front): what scene_records.pkl of an eval folder holds."""
import pickle
import sys
r = pickle.load(open(sys.argv[1], "rb"))
print(type(r), len(r))
x = r[0] if isinstance(r, list) else next(iter(r.values()))
print(type(x), (list(x)[:20] if isinstance(x, dict) else x))
