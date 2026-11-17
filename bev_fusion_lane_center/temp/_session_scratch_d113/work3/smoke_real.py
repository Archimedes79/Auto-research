import os, sys, tempfile, time, resource
R = os.environ["BFLC_ROOT"]
for sub in ("training", "training/grid", "training/data", "training/data/prep", "training/train", "evaluation"):
    sys.path.insert(0, os.path.join(R, sub))
import jmap_train as T
with tempfile.TemporaryDirectory() as tmp:
    for name, extra in (("m", []), ("m2", []), ("z", ["--no-map"])):
        t = time.time()
        a = T.parse_args(["--name", name, "--smoke", "--bs", "4", "--seed", "0"] + extra)
        T.train(a, runs=tmp, ckpt_every=500, smoke_k=512)
        print(name, "20 steps K 512 bs 4 on the CPU: %.1f s, peak RSS %.0f MB" % (time.time() - t, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024))
