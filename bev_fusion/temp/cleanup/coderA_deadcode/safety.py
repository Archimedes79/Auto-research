"""scratch safety check of the dead-code removal: python safety.py <bev_fusion root of the code under test> <out.pt> <label=config.json[=ckpt.pt]> ...

For every config: build model + loss with train.build, the ordered state_dict rows (key, shape, dtype), an eval forward on a seeded synthetic
cached batch (random init), one train-mode forward + BEVLoss on a synthetic GT (loss terms and per-parameter gradient norms), and, with a
checkpoint, strict load of its model.* / loss.* tensors (missing / unexpected keys) and an eval forward with those weights.
"""
import json
import os
import sys

import torch

root, out_path = os.path.abspath(sys.argv[1]), sys.argv[2]
specs = [a.split("=") for a in sys.argv[3:]]
torch.set_num_threads(2)
for sub in ("training", "tests/support", "tests"):
    sys.path.insert(0, os.path.join(root, sub))
import losses as L  # noqa: E402
import synthetic_world as sw  # noqa: E402
import train as T  # noqa: E402
from lift_pd import Grid  # noqa: E402
from test_slots import batch_for  # noqa: E402

print("code under test:", os.path.dirname(T.__file__), flush=True)
DTS = (-0.5, 0.0, 0.5)


def flat(tree, prefix=""):
    out = {}
    if torch.is_tensor(tree):
        out[prefix] = tree.detach().float().cpu()
    elif isinstance(tree, dict):
        for k, v in tree.items():
            out.update(flat(v, f"{prefix}/{k}"))
    elif isinstance(tree, (list, tuple)):
        for i, v in enumerate(tree):
            out.update(flat(v, f"{prefix}[{i}]"))
    return out


results = {}
grid = Grid()
scene = L._demo_scene(sw)
gts = {dt: [L.synthetic_gt(scene.step(dt) if dt else scene, grid)] for dt in DTS}
for spec in specs:
    label, cfg_path = spec[0], spec[1]
    ck_path = spec[2] if len(spec) > 2 else None
    with open(cfg_path) as f:
        cfg = T.TrainConfig(**{k: v for k, v in json.load(f).items() if not k.startswith("_") and k != "split"})      # "_..." keys are comments; "split" only names the scene lists (splits.resolve), no model field
    torch.manual_seed(0)
    model, loss, cached = T.build(cfg)
    mc = model.cfg
    res = {"rows": [(k, tuple(v.shape), str(v.dtype)) for k, v in model.state_dict().items()],
           "loss_rows": [(k, tuple(v.shape)) for k, v in loss.state_dict().items()],
           "param_order": [n for n, _ in model.named_parameters()], "n_params": sum(p.numel() for p in model.parameters())}

    def batch():
        b = batch_for(mc, B=1, Fn=cfg.n_cam)
        if mc.lane_branch:
            b["lane_img"] = torch.randint(0, 256, (1, 3, *b["hw"]), dtype=torch.uint8, generator=torch.Generator().manual_seed(5))
        return b
    step = lambda: (torch.full((1,), 1 / 12), torch.tensor([8.0]), torch.tensor([0.05]))      # noqa: E731  the ego step between two calls
    to_float = lambda out_: {dt: {k: ({a: b.float() for a, b in v.items()} if isinstance(v, dict) else v.float()) for k, v in p.items()} for dt, p in out_.items()}      # noqa: E731

    def seq_eval():
        """two calls, the second continues from the state of the first (memories written, then read): outputs and the new state"""
        with torch.no_grad():
            r1 = model(batch(), dts=DTS)
            r2 = model(batch(), dts=DTS, state=r1["state"], step=step())
        return flat(r2["out"]), flat(r2["state"])
    model.eval()
    with torch.no_grad():
        res["eval_out"] = flat(model(batch(), dts=DTS)["out"])
    res["seq_out"], res["seq_state"] = seq_eval()
    model.train()
    torch.manual_seed(1)
    r = model(batch(), dts=DTS)
    cast = to_float(r["out"])
    out = loss(cast, gts, 1.0)
    res["loss"] = {k: float(v) for k, v in out.items() if torch.is_tensor(v) and v.numel() == 1}
    out["total"].backward()
    res["grad_norm"] = {n: float(p.grad.norm()) if p.grad is not None else None for n, p in model.named_parameters()}
    # training over two calls with the state attached (BPTT through the memory write / read): loss on the second call, gradient norms
    model.zero_grad()
    loss.zero_grad()
    torch.manual_seed(2)
    r1 = model(batch(), dts=DTS)
    r2 = model(batch(), dts=DTS, state=r1["state"], step=step())
    out2 = loss(to_float(r2["out"]), gts, 1.0)
    res["seq_loss"] = {k: float(v) for k, v in out2.items() if torch.is_tensor(v) and v.numel() == 1}
    out2["total"].backward()
    res["seq_grad_norm"] = {n: float(p.grad.norm()) if p.grad is not None else None for n, p in model.named_parameters()}
    del r1, r2, out2
    if ck_path:
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        msd = {k[len("model."):]: v for k, v in ck["state"].items() if k.startswith("model.")}
        mi = model.load_state_dict(msd, strict=False)
        li = loss.load_state_dict({k[len("loss."):]: v for k, v in ck["state"].items() if k.startswith("loss.")}, strict=False)
        res["ckpt_missing"], res["ckpt_unexpected"] = sorted(mi.missing_keys), sorted(mi.unexpected_keys)
        res["ckpt_loss_missing"], res["ckpt_loss_unexpected"] = sorted(li.missing_keys), sorted(li.unexpected_keys)
        res["ckpt_state_keys"] = sorted(ck["state"])
        model.eval()
        with torch.no_grad():
            res["ckpt_out"] = flat(model(batch(), dts=DTS)["out"])
        res["ckpt_seq_out"], res["ckpt_seq_state"] = seq_eval()
    results[label] = res
    print(f"{label}: {len(res['rows'])} tensors, {res['n_params'] / 1e6:.2f} M params, total loss {res['loss']['total']:.6f}", flush=True)
    del model, loss, r, cast, out
torch.save(results, out_path)
