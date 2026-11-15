"""ms per frame of the forward (batch 1, float32): the champion next to the original modules as the evaluation runs them (B's QGraph to its sections +
the junction model `JEndModel`, which runs the pyramid and the node head a second time), in alternating blocks so that a drift of the machine cancels.

    bash tools/env/_py14.sh training/lanegraph/scripts/lg_timing.py --seed 0 [--device cuda] [--frames 8] [--rounds 6]
The GPU only when it is idle (nothing else on it); the read-out (numpy, CPU) is timed separately.
"""
import argparse
import dataclasses
import os
import sys
import time

sys.path[:0] = [os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."), os.path.dirname(os.path.abspath(__file__))]
import numpy as np  # noqa: E402
import torch  # noqa: E402

from lg_config import CHAMPION  # noqa: E402
from lg_features import FeatureStore, split_rows  # noqa: E402
from lg_model import LaneGraphNet, per_frame  # noqa: E402
from lg_project import original_models  # noqa: E402
from lg_readout import deliver  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--rounds", type=int, default=6)
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    qg, jm = original_models(a.seed, a.device)
    nets = {k: LaneGraphNet.from_runs(a.seed, dataclasses.replace(CHAMPION, attached_ends=att)).to(a.device) for k, att in (("champion", False), ("champion+attach", True))}
    xs = [FeatureStore()([r]).to(a.device) for r in split_rows("val")[: a.frames]]
    models = {"original": lambda x: (qg(dict(x=x), upto="sections", frozen=("pyr", "nodes", "links", "sections"), gt=False), jm(x)), **nets}
    sync = torch.cuda.synchronize if a.device.startswith("cuda") else (lambda: None)
    ms = {k: [] for k in models}
    for r in range(a.rounds):
        for k in (list(models) if r % 2 == 0 else list(models)[::-1]):
            models[k](xs[0])                                                          # warm
            sync()
            t0 = time.perf_counter()
            for x in xs:
                models[k](x)
            sync()
            ms[k].append(1000 * (time.perf_counter() - t0) / len(xs))
    for k, v in ms.items():
        d = np.array(v) - np.array(ms["original"])
        print(f"{k:16s} forward {np.median(v):8.1f} ms per frame  (minus original: mean {d.mean():+.1f} ms over {a.rounds} rounds, min {d.min():+.1f}, max {d.max():+.1f})")
    outs = [per_frame(nets["champion"](x))[0] for x in xs]
    t0 = time.perf_counter()
    for lanes, pairs in outs:
        deliver(lanes, pairs, CHAMPION)
    print(f"read-out (numpy, CPU) {1000 * (time.perf_counter() - t0) / len(outs):.1f} ms per frame; device {a.device}, threads {torch.get_num_threads()}")


if __name__ == "__main__":
    main()
