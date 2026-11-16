import pickle, sys
c = pickle.load(open(sys.argv[1], "rb"))
print(type(c), (list(c.keys())[:12] if isinstance(c, dict) else len(c)))
if isinstance(c, dict):
    for k, v in c.items():
        print(k, type(v).__name__, (len(v) if hasattr(v, "__len__") else v))
        if isinstance(v, list) and v:
            e = v[0]
            print("   first:", type(e).__name__, (list(e.keys()) if isinstance(e, dict) else e))
            if isinstance(e, dict):
                for kk, vv in e.items():
                    print("      ", kk, type(vv).__name__, getattr(vv, "shape", None) if hasattr(vv, "shape") else (len(vv) if hasattr(vv, "__len__") else vv))
elif isinstance(c, list):
    e = c[0]
    print(type(e).__name__, (list(e.keys()) if isinstance(e, dict) else e))
