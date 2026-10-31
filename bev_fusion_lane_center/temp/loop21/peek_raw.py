import os, sys, pickle
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "loop20"))
import e255_matched as E
E.load("fast_recall")
S = E.S
r = S["raw"][100]
print(type(r), list(S.keys()))
if isinstance(r, dict):
    for k, v in r.items():
        print(k, getattr(v, "shape", type(v)), getattr(v, "dtype", ""))
