"""Loop 16b: route attribute along the edges. Every edge of a token file is cut into pieces of at most STEP metres; the new points get the linearly interpolated token, the position on the edge and a cross-section (same structure as the input file).
The graph itself (pieces = vertex paths) keeps its geometry; only the points at which the route head is asked are denser. Usage: p16_tok_resample.py <src> <dst> [step]"""
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p16_tokens import cross_section  # noqa: E402

D = os.path.expanduser("~/exp_gcar_b123/p10")


def resample_item(it, step):
    xy = it["xy"].astype(np.float64)
    tok = it["tok"].astype(np.float32)
    has = it["has"].astype(bool)
    keep = np.asarray(it["keep"], np.float32) if "keep" in it else np.full(len(xy), 0.5, np.float32)
    X, T, H, K = [xy[i] for i in range(len(xy))], [tok[i] for i in range(len(xy))], [bool(h) for h in has], [float(k) for k in keep]
    inner = {}
    pieces = []
    for path in it["pieces"]:
        out = [path[0]]
        for a, b in zip(path[:-1], path[1:]):
            key = (min(a, b), max(a, b))
            if key not in inner:
                lo, hi = key
                ln = float(np.linalg.norm(xy[hi] - xy[lo]))
                m = int(np.ceil(ln / step)) - 1
                ids = []
                for j in range(1, m + 1):
                    w = j / (m + 1)
                    X.append(xy[lo] + w * (xy[hi] - xy[lo]))
                    if has[lo] and has[hi]:
                        T.append((1 - w) * tok[lo] + w * tok[hi])
                        H.append(True)
                    elif has[lo] or has[hi]:
                        T.append(tok[lo] if has[lo] else tok[hi])
                        H.append(True)
                    else:
                        T.append(np.zeros_like(tok[lo]))
                        H.append(False)
                    K.append((1 - w) * float(keep[lo]) + w * float(keep[hi]))
                    ids.append(len(X) - 1)
                inner[key] = ids
            ids = inner[key]
            out += (ids if a < b else ids[::-1]) + [b]
        pieces.append(out)
    V = np.asarray(X, np.float32).reshape(-1, 2)
    edges = sorted({(int(min(a, b)), int(max(a, b))) for p in pieces for a, b in zip(p[:-1], p[1:])})
    new = dict(it)
    new.update(xy=V, tok=np.asarray(T, np.float16).reshape(len(V), -1), has=np.asarray(H, bool), keep=np.asarray(K, np.float32), pieces=[list(map(int, p)) for p in pieces], edges=edges, cp=cross_section(V, edges))
    return new


def main():
    src, dst = sys.argv[1], sys.argv[2]
    step = float(sys.argv[3]) if len(sys.argv) > 3 else 4.0
    z = pickle.load(open(f"{D}/{src}.pkl", "rb"))
    n0 = np.mean([len(it["xy"]) for it in z["items"]])
    z["items"] = [resample_item(it, step) for it in z["items"]]
    n1 = [len(it["xy"]) for it in z["items"]]
    z["stats"] = dict(z.get("stats", {}), resample_step=step)
    pickle.dump(z, open(f"{D}/{dst}.pkl", "wb"))
    print(f"written {dst}: {len(z['items'])} frames, vertices per frame {n0:.0f} -> {np.mean(n1):.0f} (max {max(n1)}, over 320: {np.mean(np.asarray(n1) > 320):.3f})", flush=True)


if __name__ == "__main__":
    main()
