import pickle, os, numpy as np
ROOT = os.environ["BFLC_ROOT"]
d = pickle.load(open(os.path.join(ROOT, "temp", "e275", "pred_e273_p_s0.pkl"), "rb"))
print(type(d), len(d))
x = d[0]
for k, v in x.items():
    print(k, type(v), getattr(v, "shape", None) if hasattr(v, "shape") else (v if not isinstance(v, (list, dict)) else len(v)))
