"""Equivalence check of a refactor of training/ (used for the five-layer build of 2026-11-01, training/architecture.md): runs a
checkpoint on n val frames in eval mode and saves every forward output, or compares them with the saved reference.

    EQUIV_CODE=<old training/ snapshot> EQUIV_DEVICE=cpu python tests/equivalence.py save [--n 4]     before: the old code
    EQUIV_DEVICE=cpu python tests/equivalence.py compare                                               after: the live tree;
                                                                     max abs difference per output, exit 1 above 1e-5 or on other state keys
    EQUIV_DEVICE=cuda python tests/equivalence.py repeat --n 6       the noise floor: the same code twice (GPU scatter order)
Environment: EQUIV_RUN (run folder, default runs/t10_dual_s0), EQUIV_CKPT (default ckpt_00003000.pt), EQUIV_REF (default
temp/equivalence/ref.pt), EQUIV_CODE (training code to import, default the live training/), EQUIV_DEVICE (cpu is deterministic; on the
GPU the same code differs from itself by up to 3e-3 in the maps). An old snapshot: git archive <commit> bev_fusion/training | tar -x -C
<dir> --strip-components=2. Needs nuScenes trainval, the feature caches and the checkpoint (minutes on the CPU, about 8 GB).
"""
import argparse
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
CODE = os.environ.get("EQUIV_CODE", os.path.join(ROOT, "training"))     # the training code to run: the live tree or an old snapshot
DEV = os.environ.get("EQUIV_DEVICE", "cuda")                             # cpu: deterministic (no GPU scatter-order noise)
sys.path.insert(0, CODE)
os.chdir(ROOT)
import train as T  # noqa: E402
from feature_cache import collate_cached  # noqa: E402

REF = os.environ.get("EQUIV_REF", os.path.join(ROOT, "temp", "equivalence", "ref.pt"))


def flat(res):
    out = {"features": res["features"]}
    for dt, p in res["out"].items():
        for k, v in p.items():
            if isinstance(v, dict):
                for a, b in v.items():
                    if torch.is_tensor(b):
                        out[f"{dt}/{k}/{a}"] = b
            elif torch.is_tensor(v):
                out[f"{dt}/{k}"] = v
    return {k: v.detach().float().cpu() for k, v in out.items()}


def run(n):
    run_dir = os.environ.get("EQUIV_RUN", "runs/t10_dual_s0")
    cfg = T.read_config(os.path.join(run_dir, "config.json"))
    model, loss, cached = T.build(cfg)
    ck = torch.load(os.path.join(run_dir, os.environ.get("EQUIV_CKPT", "ckpt_00003000.pt")), map_location="cpu", weights_only=False)
    m = model.load_state_dict({k[6:]: v for k, v in ck["state"].items() if k.startswith("model.")}, strict=False)
    assert not m.unexpected_keys and all(k.startswith("camera.") for k in m.missing_keys), m
    model = model.to(DEV).eval()
    val = T.make_datasets(cfg)[1]
    idx = [int(i * (len(val) - 1) / max(n - 1, 1)) for i in range(n)]
    outs = []
    with torch.no_grad():
        for i in idx:
            b = T.to_device(collate_cached([val[i]]), DEV)
            outs.append(flat(model(b, dts=(-0.5, 0.0, 0.5))))
    return idx, outs, sorted(model.state_dict())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["save", "compare", "repeat"])
    ap.add_argument("--n", type=int, default=20)
    a = ap.parse_args()
    torch.manual_seed(0)
    if a.cmd == "repeat":                              # the noise floor: the same code twice (GPU scatter / index_add order)
        _, o1, _ = run(a.n)
        _, o2, _ = run(a.n)
        worst = {}
        for x, y in zip(o1, o2):
            for k in x:
                worst[k] = max(worst.get(k, 0.0), (x[k] - y[k]).abs().max().item())
        print("repeat: max abs diff per output:", {k: f"{v:.1e}" for k, v in sorted(worst.items(), key=lambda kv: -kv[1])[:8]})
        return
    if a.cmd == "save":
        idx, outs, keys = run(a.n)
        os.makedirs(os.path.dirname(os.path.abspath(REF)), exist_ok=True)
        torch.save({"idx": idx, "outs": outs, "keys": keys}, REF)
        print("saved", len(outs), "frames,", len(outs[0]), "outputs per frame,", len(keys), "state keys ->", REF)
        return
    ref = torch.load(REF)
    idx, outs, keys = run(len(ref["idx"]))
    assert idx == ref["idx"]
    if keys != ref["keys"]:
        print("state_dict keys differ:", sorted(set(keys) ^ set(ref["keys"]))[:10])
        sys.exit(1)
    worst = {}
    for o, r in zip(outs, ref["outs"]):
        for k in r:
            if k not in o:
                print("missing output", k)
                sys.exit(1)
            worst[k] = max(worst.get(k, 0.0), (o[k] - r[k]).abs().max().item())
    new = sorted(set(outs[0]) - set(ref["outs"][0]))
    bad = {k: v for k, v in worst.items() if v > 1e-5}
    print(f"{len(idx)} frames, {len(worst)} outputs compared, max abs diff {max(worst.values()):.2e}; new outputs: {new}")
    if bad:
        print("ABOVE 1e-5:", bad)
        sys.exit(1)
    print("equivalent within 1e-5; state_dict keys identical")


if __name__ == "__main__":
    main()
