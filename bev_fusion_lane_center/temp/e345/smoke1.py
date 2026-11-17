import os, sys, time, json, numpy as np, torch
sys.path.insert(0, os.path.join(os.environ["BFLC_ROOT"], "evaluation"))
import jprune_probe as P
torch.set_num_threads(2)
t = time.time()
sw = P.Sweep("e312_p_s0", "cpu", "temp/e345/out_smoke", None, 16, 16, workers=0)
os.makedirs("temp/e345/out_smoke", exist_ok=True)
print("init", time.time() - t)
c = sw.calib
for k in ("pyr.e1", "jm.blocks.1.f.2", "je.blocks.0.ffn.1", "je.pair.1", "je.fcat", "jm.head1"):
    print(k, c.mean[k].shape, float(c.s[k].min()), float(c.s[k].max()))
print({k: tuple(v.shape) for k, v in list(c.head.items())[:3]})
print("fcat cols 145/146 s:", c.s["je.fcat"][[145, 146]], c.mean["je.fcat"][[145, 146]])
t = time.time()
r, frames = P.run_pass(sw.model, sw.val, "cpu", sw.half, keep_frames=True)
print("null pass", time.time() - t, r["pair"], P.six(np.sum(r["counts"], 0)), np.sum(r["kp"], 0))
sw.do("null", {}); sw.do("null2", {})
sw.do("p007", {"pyr.d0": 0.25})
sw.do("p003", {"je.blocks.0.skip": 1.0})
sw.do("pHead", {"je.blocks.1.att.h2": 1.0})
sw.do("pPaint", {"je.paint": 1.0})
sw.do("pStream", {"jm.stream": 0.5})
sw.plumbing()
print(json.dumps(sw.rec["controls"]))
