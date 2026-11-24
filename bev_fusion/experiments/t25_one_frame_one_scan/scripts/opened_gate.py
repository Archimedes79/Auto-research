"""t25 opened gate (card section 1 "Added 2026-11-24" and section 3): did the radar memory of an A2 checkpoint move away from the identity?

    python experiments/t25_one_frame_one_scan/scripts/opened_gate.py runs/t25_A2_s0 [ckpt_file]     (from bev_fusion/, CPU, a few seconds)

The gate is TRUE when RMS(rad_mix.weight) >= 0.10 x RMS(rad_temporal.fuse.0.weight), both from the EMA weights of the newest checkpoint of the run
(the weight rad_mix starts at exactly 0). A weak proxy, written before the run: it says only that the block moved, not that the output uses it.
"""
import glob
import os
import sys

import torch


def rms(t):
    return float(t.float().pow(2).mean().sqrt())


def find(sd, name):
    """the tensor of a state dict whose key ends with `name` (the checkpoint nests the model weights under a prefix)"""
    hits = [k for k in sd if k.endswith(name)]
    if len(hits) != 1:
        raise SystemExit(f"opened_gate: {len(hits)} keys end with {name!r}: {hits[:4]}")
    return sd[hits[0]]


def weights(ck):
    for key in ("ema", "ema_model", "model_ema", "model", "state_dict"):
        if isinstance(ck, dict) and key in ck and isinstance(ck[key], dict):
            return key, ck[key]
    return "top level", ck


def main():
    run = sys.argv[1]
    path = sys.argv[2] if len(sys.argv) > 2 else sorted(glob.glob(os.path.join(run, "ckpt_*.pt")))[-1]
    ck = torch.load(path, map_location="cpu", weights_only=False)
    src, sd = weights(ck)
    mix = rms(find(sd, "rad_mix.weight"))
    ref = rms(find(sd, "rad_temporal.fuse.0.weight"))
    print(f"checkpoint {path}; weights from '{src}'; RMS rad_mix.weight {mix:.5f}; RMS rad_temporal.fuse.0.weight {ref:.5f}; ratio {mix / ref:.4f}; "
          f"opened gate (ratio >= 0.10): {mix >= 0.10 * ref}")


if __name__ == "__main__":
    main()
