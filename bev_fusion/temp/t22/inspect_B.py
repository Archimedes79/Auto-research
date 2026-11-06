"""scratch: stage-B run t22_B_s0 - log summary against the kill criteria and a check that the checkpoint holds fine-tuned student weights"""
import json, math, sys
import torch
torch.set_num_threads(2)
run = "runs/t22_B_s0"
rows = [json.loads(l) for l in open(f"{run}/log.jsonl") if l.strip()]
print("log rows", len(rows), "keys of the last row:", sorted(rows[-1].keys())[:60])
steps = [r for r in rows if "total" in r and "step" in r]
nonfin = [r["step"] for r in steps if not math.isfinite(r["total"])]
print("steps with a total:", len(steps), "non-finite:", len(nonfin))
for k in ("gpu_gb", "student_anchor", "grad.student", "sec_per_step", "step_s"):
    v = [r[k] for r in steps if k in r and isinstance(r[k], (int, float))]
    if v:
        print(f"{k}: n={len(v)} first={v[0]:.3f} last={v[-1]:.3f} min={min(v):.3f} max={max(v):.3f}")
vals = [r for r in rows if any(k.startswith("val") for k in r)]
for r in vals:
    print("val row:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k in ("step",) or k.startswith("val")})
# the checkpoint: student weights present and different from the A2 stage-A file
ck = torch.load(f"{run}/ckpt_00001500.pt", map_location="cpu", weights_only=False)
print("ckpt keys:", list(ck.keys())[:12])
for part in ("state", "ema"):
    sdp = ck[part]
    print(part, "tensors:", len(sdp), "student ones:", sum(1 for k in sdp if "student" in k))
sd = ck["state"]
sk = [k for k in sd if "student" in k]
print("student tensors in the checkpoint state:", len(sk), sk[:3])
a2 = torch.load("runs/t22_A2_s0/ckpt_00006108.pt", map_location="cpu", weights_only=False)
print("A2 ckpt keys:", list(a2.keys())[:12])
a2sd = a2["ema"] if a2.get("ema") else a2["state"]
print("A2 tensors:", len(a2sd), list(a2sd)[:3], "(ema used)" if a2.get("ema") else "(state used)")
print("extra of the B ckpt:", str(ck.get("extra"))[:300])
# compare by suffix match
import itertools
diffs = []
for k in sk:
    suf = k.split("student.", 1)[1] if "student." in k else k
    cands = [m for m in a2sd if m == suf or m.endswith(suf)]
    if cands and sd[k].shape == a2sd[cands[0]].shape and sd[k].dtype.is_floating_point:
        diffs.append((k, float((sd[k].float() - a2sd[cands[0]].float()).abs().max())))
print("compared", len(diffs), "student tensors against A2; max abs diff overall", max(d for _, d in diffs) if diffs else None, "; tensors with diff > 0:", sum(1 for _, d in diffs if d > 0))
print("largest:", sorted(diffs, key=lambda x: -x[1])[:3])
