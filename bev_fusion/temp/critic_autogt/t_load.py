import glob, time, numpy as np, os
from autogt import cli
fs = sorted(glob.glob("data/autogt/fefd519974/*.npz"))[:20]
d = os.path.dirname(fs[0]); toks = [os.path.basename(f)[:-4] for f in fs]
t = time.time(); [cli.load_keyframe(d, k) for k in toks]; a = (time.time() - t) / len(toks)
def fast(sig_dir, token):
    with np.load(os.path.join(sig_dir, token + ".npz"), allow_pickle=True) as z:
        out = {}
        for k in z.files:
            if k == "meta": continue
            v = z[k]; out[k] = v.astype(np.float32) if v.dtype == np.float16 else v
        out["meta"] = z["meta"].item()
    return out
t = time.time(); r = [fast(d, k) for k in toks]; b = (time.time() - t) / len(toks)
x = cli.load_keyframe(d, toks[0]); y = fast(d, toks[0])
print(f"load_keyframe {a*1000:.0f} ms, single-read variant {b*1000:.0f} ms; identical: {all(np.array_equal(x[k], y[k]) for k in x if k != 'meta')}")
