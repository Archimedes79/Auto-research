"""scratch (L4 check, 2026-11-08): bev_suite with the stride-4 lane branch (L4) dropped after the checkpoint load.

    python temp/t22/lane_off/suite_lane_off.py RUN [bev_suite arguments as usual]

The adapter's load_run is wrapped: the run is loaded as its config says (lane branch res4, so the checkpoint fits), then `model.lane = None`. The forward of training/model.py reads
`if self.lane is not None: ... cam = cam + lane`, so with None it computes the network without the branch (the camera BEV is the temporal block's output as it is). The rest of the
network was trained WITH the branch: this is the branch's contribution as it stands, not a network trained without it. It prints one line per load ("L4 dropped: ... parameters").
No frozen file is touched (the wrapper patches the module attribute in this process only). `install()` is also what test_patch.py checks."""
import os
import runpy
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)


def install():
    from evaluation import bev_adapter
    orig = bev_adapter.load_run

    def load_run_l4off(*a, **k):
        cfg, model, loss, step, path = orig(*a, **k)
        assert getattr(model, "lane", None) is not None, "the run has no lane branch to drop"
        n = sum(p.numel() for p in model.lane.parameters())
        model.lane = None
        assert model.lane is None and not any(name.startswith("lane.") for name, _ in model.named_parameters())
        print(f"L4 dropped: {n} parameters of the lane branch are out of the forward (step {step}, {path})", flush=True)
        return cfg, model, loss, step, path

    bev_adapter.load_run = load_run_l4off
    return bev_adapter


if __name__ == "__main__":
    install()
    sys.argv = [os.path.join(ROOT, "evaluation", "bev_suite.py")] + sys.argv[1:]
    runpy.run_path(sys.argv[0], run_name="__main__")
