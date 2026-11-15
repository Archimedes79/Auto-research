import contextlib, io, os, sys, tempfile
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path[:0] = [os.path.join(ROOT, "training"), os.path.join(ROOT, "tests", "support"), ROOT, os.path.join(ROOT, "..")]
import train as T
import toy_train as TT
real = T.run_clip
T.run_clip = lambda *a, **k: real(*a, **{kk: vv for kk, vv in k.items() if kk not in ("step", "micro")})
seen = []
def forward(model, loss, batch, *a, **k):
    j = batch.get("img_jitter")
    seen.append(None if j is None else (j["u"].flatten().tolist(), j["noise_seed"]))
    return TT.toy_forward(model, loss, batch, *a, **k)
T.build, T.forward_loss, T.make_loader, T.val_subset = (lambda cfg: (TT.ToyNet(), TT.ToyLoss(), True)), forward, (lambda ds, cfg, train, workers=None, generator=None: ds), (lambda ds, n: ds)
tmp = tempfile.mkdtemp()
cfg = T.TrainConfig(name="x", out_root=tmp, steps=8, accum=2, warmup=2, log_every=1, eval_every=0, save_every=4, keep=3, device="cpu", workers=0, flip_aug=0.0, seq_len=1, seed=0, lr=1e-2, grad_clip=0.5, img_jitter=0.5, img_jitter_from=0.25)
data = TT.toy_batches(20)
buf = io.StringIO()
try:
    with contextlib.redirect_stdout(buf):
        T.train(cfg, datasets=(data, data))
except BaseException as e:
    print("EXC", type(e).__name__, e)
print(len(seen), seen[:6])
print(buf.getvalue()[-600:])

import traceback
def run(name, **kw):
    seen2 = []
    def fwd(model, loss, batch, *a, **k):
        j = batch.get("img_jitter")
        seen2.append(None if j is None else (j["u"].flatten().tolist(), j["noise_seed"]))
        return TT.toy_forward(model, loss, batch, *a, **k)
    T.forward_loss = fwd
    base = dict(name=name, out_root=tmp, steps=8, accum=2, warmup=2, log_every=1, eval_every=0, save_every=4, keep=3, device="cpu", workers=0, flip_aug=0.0, seq_len=1, seed=0, lr=1e-2, grad_clip=0.5, img_jitter=0.5, img_jitter_from=0.25)
    data = TT.toy_batches(20)
    with contextlib.redirect_stdout(io.StringIO()):
        T.train(T.TrainConfig(**dict(base, **kw)), datasets=(data, data))
    return seen2
a, b, c = run("tj_a"), run("tj_b"), run("tj_c", seed=1)
try:
    print(len(a), len(b), len(c))
    print(all(a[i][0] != a[i + 1][0] for i in range(4, 15, 2)))
    print(a == b, a[4:] != c[4:], all(x[0] != y[0] for x, y in zip(a[4:], c[4:])))
except Exception:
    traceback.print_exc()
