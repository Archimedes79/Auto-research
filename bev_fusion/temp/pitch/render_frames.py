"""The same validation frames rendered for two or more runs with the suite's overview page (camera with GT | GT bird's-eye view | prediction |
error maps; grey right, RED false, BLUE missed). CPU, one process, the frames are the suite's deterministic spread so every run gets the same ones.
Usage (bev_fusion folder, gcar python): python temp/pitch/render_frames.py <n frames> <out dir> <run> [<run> ...]"""
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "2")
sys.path[:0] = [".", "..", "../autoresearch", "evaluation", "training"]

n, out_root, runs = int(sys.argv[1]), sys.argv[2], sys.argv[3:]
from evalsuite import EvalSuite  # noqa: E402
from evalsuite.sampling import Sampler  # noqa: E402
from evaluation.bev_adapter import build  # noqa: E402

for run in runs:
    source, subject, task = build(f"runs/{run}", device="cpu", split="val", workers=0, ckpt_dir=None, control="", prior=0)
    suite = EvalSuite(source, subject, task, out=os.path.join(out_root, run), state="reset")
    refs = Sampler(source, 0).spread(n)
    for ref in refs:
        print(run, ref.key, flush=True)
        suite.qual.frame(ref, ("overview",), out_dir=os.path.join(out_root, run), history="none", state="reset")
    suite.close()
print("done")
