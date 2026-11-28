"""Compute time and memory of a lane network as the evaluation reports them: the metric `cost` of `adapter.py` (in the evaluator hash).

The predict step (`lane_predict.measure_cost`) runs the whole path - B' (pyramid, node head, the top-K tokens) and the token head with its layers - one frame at a time on the GPU (bf16 autocast, CUDA events: `jtiming`)
and writes into the dump: every frame's `ms`, and in the meta `cost`:
  peak_MB    the allocator's peak during these frames, MiB: the weights and the activations of one frame (the cached foundation features and the CUDA context are not part of it)
  params_M   the parameters of that path, millions
  quiet      1 = no other compute process on the GPU, and the host and the GPU idle before and after (`jtiming.quiet_verdict`); else 0
  host       the line that says so
Memory and parameters compare between any two runs on the same card; a time compares only between runs that are both quiet. A dump without a measurement (a CPU dump of a test) gives nan."""
import numpy as np
from Evaluation import Metric

FRAME_KEYS = ("ms", "peak_MB", "params_M", "quiet")                 # what the adapter hands over per frame (the run's values repeated on every frame)
KEYS = ("ms_median", "ms_p90", "peak_MB", "params_M", "quiet")


class LaneCost(Metric):
    name, headline = "cost", ("ms_median", "peak_MB", "params_M")
    better = {k: ("high" if k == "quiet" else "low") for k in KEYS}

    def frame(self, sample, pred):
        c = pred.info.get("cost") or {}
        return {k: np.array([np.nan if c.get(k) is None else float(c[k])]) for k in FRAME_KEYS}

    def reduce(self, counts, keys=None):
        a = {}
        for k in FRAME_KEYS:
            x = np.concatenate([np.atleast_1d(np.asarray(c[k], float)) for c in counts]) if counts else np.zeros(0)
            a[k] = x[np.isfinite(x)]
        of = lambda k, f: float(f(a[k])) if len(a[k]) else float("nan")
        out = {"ms_median": of("ms", np.median), "ms_p90": of("ms", lambda x: np.percentile(x, 90)), "peak_MB": of("peak_MB", np.max), "params_M": of("params_M", np.max),
               "quiet": of("quiet", np.min)}
        return {k: out[k] for k in (keys or out) if k in out}
