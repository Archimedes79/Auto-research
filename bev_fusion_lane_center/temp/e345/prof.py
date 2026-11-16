import os, sys, time, torch
sys.path.insert(0, os.path.join(os.environ["BFLC_ROOT"], "evaluation"))
import jprune_probe as P
import qendnet as QEN
torch.set_num_threads(2)
m, _ = P.JEV.load_model("e312_p_s0", "cpu")
cache = P.build_cache("val", workers=0, max_frames=16)
out, b = next(iter(P.forward_batches(m, cache, "cpu")))
def t(fn, n=3):
    fn(); t0 = time.time()
    for _ in range(n): fn()
    return (time.time() - t0) / n
n = len(b["token"])
print("end_counts per frame ms", 1000 * t(lambda: [QEN.end_counts(P._slice(out, i), P._slice(b, i), 0.1) for i in range(n)]) / n)
print("match_keypoints per frame ms", 1000 * t(lambda: [QEN.match_keypoints(P._slice(out, i), P._slice(b, i)) for i in range(n)]) / n)
print("pair_candidates per frame ms", 1000 * t(lambda: [P.pair_candidates(out, b, i) for i in range(n)]) / n)
print("kp_error per frame ms", 1000 * t(lambda: [P.kp_error(out, b, i) for i in range(n)]) / n)
print("frames_of per batch ms", 1000 * t(lambda: P.frames_of(out, b)))
print("end_counts on the batch ms", 1000 * t(lambda: QEN.end_counts(out, b, 0.1)))
