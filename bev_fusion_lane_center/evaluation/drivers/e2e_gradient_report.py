"""Gradient report of the lane-graph builder: which parts of the network receive gradient from the training loss, and does the gradient reach the input features?

    wsl bash tools/env/_py14.sh e2e_gradient_report.py --run e123_D7_r3_f1 [--n 2] [--json out.json]
    wsl bash tools/env/_py14.sh e2e_gradient_report.py --selftest

One forward + backward of the registered builder (config from its run card, weights from its checkpoint) on real VAL frames, on the CPU in float32 (two threads: the
box is shared), with the loss of `p16_loss.total_loss` exactly as the trainer calls it. It reports per module group the parameters, those with a nonzero gradient and the
share of the total gradient norm, and the gradient on the 42 input channels (39 frozen GaussianCaR channels, dense-head heat, dense-head direction): that is the gradient a
trainable front end (D26) would receive. It measures only what the loss reaches. What the loss never sees (the route head trains on stored token files, the assembly
runs after the network) is a property of the architecture, named in training/architecture.md, not of this report.
Use: before and after every replacement of a rule step by a differentiable one, the groups that were silent must light up (the lane analogue of bev_fusion's tests/test_slots.py)."""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

GROUPS = [  # (group, name prefixes of Builder.named_parameters)
    ("stem (conv pyramid)", ("stem.",)),
    ("pyramid projections", ("pr.",)),
    ("seeds (lattice read-out, snap, direction)", ("seed_mlp.", "seed_val.", "seed_snap.", "seed_dir.", "flag_emb.")),
    ("context (attention over tips)", ("ctx.",)),
    ("ray read-out (verification, offsets, tokens)", ("geo_emb", "hp.", "read.", "read_conv.", "out_v.", "out_off.", "out_emb.", "exist.", "loi_head.")),
    ("junction head", ("junc_head.",)),
    ("child tokens", ("child.", "child_norm.", "rf_mlp.", "rf_tok.", "rf_norm.")),
    ("rescoring (stage C)", ("rc_layer.", "rc_head.")),
    ("keep gate / pair head / revisit", ("keep.", "pair.", "pair_proj.", "rev.", "rev_in.")),
    ("link head", ("link.", "conn_v.")),
]
INPUT_GROUPS = [("GaussianCaR decoder PCA (32)", 0, 32), ("GaussianCaR map logits (7)", 32, 39), ("dense-head heat (1)", 39, 40), ("dense-head direction (2)", 40, 42)]


def group_of(name):
    for g, prefixes in GROUPS:
        if name.startswith(prefixes):
            return g
    return "other"


def param_table(named_params, grad_of=group_of):
    """rows (group, n params, n tensors, tensors with a nonzero gradient, gradient L2 norm), and the names of the tensors without gradient"""
    rows, silent = {}, []
    for n, p in named_params:
        g = grad_of(n)
        r = rows.setdefault(g, dict(group=g, n=0, tensors=0, live=0, sq=0.0))
        r["n"] += p.numel()
        r["tensors"] += 1
        gr = p.grad
        if gr is not None and float(gr.abs().max()) > 0:
            r["live"] += 1
            r["sq"] += float((gr.float() ** 2).sum())
        elif p.requires_grad:
            silent.append(n)
    tot = sum(r["sq"] for r in rows.values()) or 1.0
    out = []
    for r in rows.values():
        out.append(dict(group=r["group"], params=r["n"], tensors=r["tensors"], live=r["live"], norm=r["sq"] ** 0.5, share=r["sq"] / tot))
    return out, silent


def input_table(xg):
    """mean absolute gradient per input channel group, from the gradient of the loss on the input x [B, C, H, W]"""
    ga = xg.abs().float().mean((0, 2, 3))
    return [dict(group=g, channels=b - a, mean_abs=float(ga[a:b].mean()), max_abs=float(xg[:, a:b].abs().max())) for g, a, b in INPUT_GROUPS]


def run(run_name, n, seed=0):
    import p16_builder as PB
    import p16_common as C
    import p16_data as PD
    import p16_loss as PL
    import p16_time as PT_

    torch.manual_seed(seed)
    card = json.load(open(f"{C.LADDER}/cards/{run_name}.json", encoding="utf-8-sig"))
    cfg = PB.DefaultCfg(**card["cfg"])
    net = PB.Builder(cfg, stem_from=None)
    sd = torch.load(f"{C.OUT}/e123/{run_name}/model.pt", map_location="cpu")
    miss = net.load_state_dict(sd, strict=False)
    net.float().train()
    ds = PD.BuilderData("val_clean", ks=list(range(n)), train=False, with_prev=cfg.temporal)
    b = PD.collate([ds[i] for i in range(n)])
    x = b["x"].float().requires_grad_(True)
    prior = PT_.prior_from_prev(net, b, train=True) if cfg.temporal else None
    res = net(x, gt=None, share=0.0, prior=prior)
    loss, parts = PL.total_loss(net, res, b, {})
    loss.backward()
    rows, silent = param_table(net.named_parameters())
    return dict(run=run_name, frames=n, cfg=card["cfg"], missing_keys=len(miss.missing_keys), loss=float(loss), parts={k: float(v) for k, v in parts.items() if k in ("ver", "pos", "val", "prec", "rec", "junc", "link", "dup", "aux_seed", "aux_dir", "tot")},
                params=rows, silent=silent, inputs=input_table(x.grad))


def show(rep):
    print(f"gradient report of {rep['run']} on {rep['frames']} VAL frames (CPU, float32); loss {rep['loss']:.3f}; checkpoint keys missing: {rep['missing_keys']}")
    print(f"{'module group':48s} {'params':>9s} {'tensors live':>13s} {'grad norm':>10s} {'share':>7s}")
    for r in rep["params"]:
        print(f"{r['group']:48s} {r['params']:9d} {r['live']:>6d}/{r['tensors']:<6d} {r['norm']:10.4f} {100 * r['share']:6.1f}%")
    print("tensors without any gradient:", ", ".join(rep["silent"]) if rep["silent"] else "none")
    print(f"{'input channels':48s} {'n':>9s} {'mean |grad|':>13s} {'max |grad|':>12s}")
    for r in rep["inputs"]:
        print(f"{r['group']:48s} {r['channels']:9d} {r['mean_abs']:13.3e} {r['max_abs']:12.3e}")


def selftest():
    import torch.nn as nn

    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem, self.seed_val, self.link = nn.Linear(4, 4), nn.Linear(4, 1), nn.Linear(1, 1)

        def forward(self, x):
            h = self.stem(x)
            y = self.seed_val(h)
            return self.link(y.detach())  # a rule-step-like cut: link gets gradient, everything before it only through seed_val's own loss

    m, x = Toy(), torch.randn(3, 4, requires_grad=True)
    m(x).sum().backward()
    rows, silent = param_table(m.named_parameters())
    d = {r["group"]: r for r in rows}
    assert d["link head"]["live"] == 2 and d["stem (conv pyramid)"]["live"] == 0 and "stem.weight" in silent and "seed_val.bias" in silent, (d, silent)
    assert x.grad is None or float(x.grad.abs().sum()) == 0.0  # the detach cuts the input too
    xg = torch.zeros(2, 42, 3, 3)
    xg[:, 39] = 2.0
    t = {r["group"]: r for r in input_table(xg)}
    assert t["dense-head heat (1)"]["mean_abs"] == 2.0 and t["GaussianCaR decoder PCA (32)"]["mean_abs"] == 0.0
    assert group_of("junc_head.0.weight") == "junction head" and group_of("zzz") == "other"
    print("e2e_gradient_report selftest passed")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="e123_D7_r3_f1", help="run name under runs/e123 (card in runs/e123_ladder/cards)")
    ap.add_argument("--n", type=int, default=2, help="VAL frames")
    ap.add_argument("--json", default="", help="also write the report here")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    torch.set_num_threads(2)
    rep = run(a.run, a.n)
    show(rep)
    if a.json:
        json.dump(rep, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
