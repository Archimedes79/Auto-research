"""The champion's stage-1 head (E373, `runs/e373_cb_s<seed>`) as the p_none of cached tokens: the model step that `net/prep/lane_mapassoc_data` needs for
the TRAIN frames. It lives with the trainer whose `load_model` builds the head; preprocessing gets it from its caller (D152 follow-up)."""
import torch

import jtoken as JK
import jtoken_train as JKT


def head_pnone(seed):
    """-> a function batch (`lane_ptok_data.collate`) -> p_none (B, K) numpy, of the stored head of this seed, on the CPU with 2 threads, without gradients"""
    torch.set_num_threads(2)
    m, _ = JKT.load_model(f"e373_cb_s{seed}", "cpu")

    def pnone(b):
        with torch.no_grad():
            return JK.probs(m(b))["pt"][..., 0].numpy()
    return pnone


def _selftest():
    """the run of the seed on the CPU, and p_none = column 0 of the head's `pt` (a stubbed loader and head: the real run is read by the map-association build)"""
    seen = []
    real = JKT.load_model, JK.probs

    def load(run, dev):
        seen.append((run, dev))
        return (lambda b: b), {}
    JKT.load_model, JK.probs = load, lambda t: {"pt": torch.stack([t, t + 100.0], -1)}
    try:
        x = torch.arange(6.0).reshape(2, 3).requires_grad_(True)
        out = head_pnone(1)(x)
    finally:
        JKT.load_model, JK.probs = real
    assert seen == [("e373_cb_s1", "cpu")] and out.shape == (2, 3) and (out == x.detach().numpy()).all(), (seen, out)
    print("champion_pnone selftest ok: the head of runs/e373_cb_s<seed> on the CPU, p_none = column 0, no gradient kept")


if __name__ == "__main__":
    _selftest()
