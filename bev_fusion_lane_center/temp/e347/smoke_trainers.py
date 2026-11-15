"""E347 scratch (one use): the wiring of the two trainers on real VAL-sized data on the CPU, with the run folder and the optimiser loop stubbed out (no run folder, no checkpoint, no training):
`jmaps_train.py --no-paint --r2 [--onehot]` builds the 2 m head and its data, one real batch gives a finite loss with the offset term; `jends_train.py --e2e --scales --maps <r2 maps checkpoint>` builds the pair head for the 2 m head, one real batch gives
the pair loss + the maps loss, and the offset head gets a gradient. Run through temp/e347/run_staged.sh (CUDA off).
    wsl bash temp/e347/run_staged.sh smoke_trainers.py
"""
import os
import sys

import torch

ROOT = os.environ["BFLC_ROOT"]
os.environ["BFLC_SIG"] = "1.0"
import jmaps_train as MT  # noqa: E402
import jends_train as JT  # noqa: E402
import jmaps as JM  # noqa: E402


class Stop(Exception):
    """raised by the stubbed fit after one real batch: main() stops before it saves anything"""


class FakeRun:
    def __init__(self, name):
        self.name = name

    def log(self, *a):
        print("  log:", " ".join(str(x) for x in a)[:260])

    def path(self, f):
        return os.devnull

    def mark_done(self):
        pass


def run_main(mod, argv, grad_check=None):
    saved = {}
    out = {}

    def fake_fit(model, loss_fn, run, *, steps, loader, lr, ckpt_every, seed, args, params, config):
        b = next(iter(loader))
        for p in params:
            p.requires_grad_(True)
        loss, info = loss_fn(model, b)
        loss.backward()
        out["loss"], out["info"], out["model"], out["batch"] = float(loss), info, model, b
        out["params"] = params
        print("  one real batch: loss %.4f" % float(loss), {k: round(float(v), 4) for k, v in info.items() if k in ("j_off", "j_heat", "j_zone", "j_dir", "bce", "pos_pairs", "cross", "scale_l1")})
        raise Stop()

    mod.T.Run, mod.T.fit = FakeRun, fake_fit
    sys.argv = ["x"] + argv
    try:
        mod.main()
    except Stop:
        pass
    return out


if __name__ == "__main__":
    smoke = os.path.join(ROOT, "temp", "_session_scratch_e347", "smoke")
    os.makedirs(smoke, exist_ok=True)
    base = os.path.join(ROOT, "runs", "e273_p_s0", "model.pt")
    for extra in ([], ["--onehot"]):
        print("jmaps_train.py --no-paint --r2", " ".join(extra))
        o = run_main(MT, ["--name", "smoke_r2", "--base", base, "--no-paint", "--r2", *extra, "--bs", "2", "--workers", "0", "--sigma", "1.0"])
        assert o["batch"]["jz"].shape[-1] == 50 and o["batch"]["jend"].shape[1:] == (2, 50, 50) and o["info"]["j_off"] > 0 and ("onehot" in o["model"].jm(o["model"].pyr(o["batch"]["x"].float()))) == bool(extra)
    for bad in (["--r2"], ["--no-paint", "--onehot"], ["--no-paint", "--r2", "--dirx"], ["--no-paint", "--r2", "--far-w", "1.0"]):
        try:
            run_main(MT, ["--name", "smoke_bad", "--base", base, *bad, "--bs", "2", "--workers", "0"])
            print("  NOT REFUSED:", bad)
        except (AssertionError, SystemExit) as e:
            print("  refused:", bad, "->", str(e)[:100])
    # a random-weight 2 m maps checkpoint (ctx and args as jmaps_train.py would store them), then the pair stage on it, e2e
    for ctx in ("dil+nopaint+r2", "dil+nopaint+r2+onehot"):
        torch.manual_seed(0)
        m = JM.JModel(width=48, d=64, ctx=ctx)
        m.load_base(base)
        ck = os.path.join(smoke, "maps_" + ctx.replace("+", "_") + ".pt")
        torch.save(dict(state=m.state_dict(), args=dict(ctx=ctx, d=64, seed=0, base=None, raster="legacy", sigma=1.0)), ck)
        print("jends_train.py --e2e --scales on", ctx)
        o = run_main(JT, ["--name", "smoke_p", "--maps", ck, "--e2e", "--scales", "--maps-w", "1.0", "--bs", "2", "--workers", "0"])
        m2 = o["model"]
        g = float(m2.jmodel.jm.head2.weight.grad[8:12].abs().sum())
        print("  gradient of the offset channels of head2:", g, "| pair head:", (m2.je.n1, m2.je.offset, m2.je.peak), "| keys of the maps loss:", [k for k in o["info"] if k.startswith("j_")])
        assert g > 0 and (m2.je.n1, m2.je.offset, m2.je.peak) == (50, True, "onehot" not in ctx) and o["info"]["pos_pairs"] > 0
    print("smoke_trainers ok")
