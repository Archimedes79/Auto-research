"""one-off (framework testing before the first small training, 2026-11-01): the five-layer framework on REAL data, no training.
The hurdle dual chain (runs/t10_dual_long_s0, ckpt 5030) on val frames, GPU, bf16 as in training:
 1 input contract on real batches (collate_cached of single frames and of clips)
 2 a clip of 3 keyframes through train.run_clip with the flip forced on, backward: finite loss, finite gradients, which trainable
   parameters get none
 3 dense outputs on: output contract, model.boxes() vs object_head.decode on the same maps
 4 warm: the memory carried over the clip changes the last keyframe's output (the memory is read)
 5 the precompute maps under flip: vis_cam of a flipped sample = the mirrored map of the unflipped one
Run from bev_fusion/: python temp/arch_build/dryrun.py   (needs trainval + caches; about 10 GB host memory)"""
import dataclasses
import os
import sys
import time

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "training"))
os.chdir(ROOT)
import train as T  # noqa: E402
from augment import flip_batch  # noqa: E402
from feature_cache import collate_cached  # noqa: E402
from interface import check_inputs, check_outputs  # noqa: E402
from losses import NUSC_GROUPS  # noqa: E402
from object_head import decode  # noqa: E402
from precompute import Precompute  # noqa: E402

t0 = time.time()
problems = []


def note(ok, msg):
    print(("  ok    " if ok else "  FAIL  ") + msg, flush=True)
    if not ok:
        problems.append(msg)


cfg = T.read_config("runs/t10_dual_long_s0/config.json")
cfg = dataclasses.replace(cfg, flip_aug=1.0)                      # every sample mirrored in step 2
model, loss, cached = T.build(cfg)
ck = torch.load("runs/t10_dual_long_s0/ckpt_00005030.pt", map_location="cpu", weights_only=False)
m = model.load_state_dict({k[6:]: v for k, v in ck["state"].items() if k.startswith("model.")}, strict=False)
loss.load_state_dict({k[5:]: v for k, v in ck["state"].items() if k.startswith("loss.")})
note(not m.unexpected_keys and all(k.startswith("camera.") for k in m.missing_keys), "checkpoint 5030 loads into the rebuilt model")
model, loss = model.cuda(), loss.cuda()
val = T.make_datasets(cfg)[1]
print(f"data ready ({time.time() - t0:.0f} s): {len(val)} val samples", flush=True)

# 1 input contract on real batches
b = collate_cached([val[0], val[57]])
e = check_inputs(b)
note(not e, f"check_inputs on a real batch: {e[:4] if e else 'clean'} (keys: {sorted(b)})")
clips = T.ClipDataset(val, 3)
clip = T.collate_clips([clips[0], clips[len(clips) // 2]], collate_cached)
note(all(not check_inputs(x) for x in clip), "check_inputs on every keyframe of a real clip")

# 2 a training clip with flip, bf16, backward
model.train()
model.zero_grad(set_to_none=True)
acc, finite, _, totals = T.run_clip(model, loss, clip, 0.5, "cuda", cfg.amp, cfg, backward=True, augment=True)
note(finite and all(torch.isfinite(torch.tensor(totals))), f"run_clip (3 keyframes, flip on, amp {cfg.amp} as the runs): finite, totals {[round(x, 3) for x in totals]}")
clip_b = T.collate_clips([clips[0], clips[len(clips) // 2]], collate_cached)
model.zero_grad(set_to_none=True)
_, finite_b, _, totals_b = T.run_clip(model, loss, clip_b, 0.5, "cuda", "bf16", cfg, backward=True, augment=True)
note(finite_b, f"run_clip in bf16 (the splat index fix): finite, totals {[round(x, 3) for x in totals_b]}")
model.zero_grad(set_to_none=True)
_, _, _, _ = T.run_clip(model, loss, T.collate_clips([clips[0], clips[len(clips) // 2]], collate_cached), 0.5, "cuda", cfg.amp, cfg,
                        backward=True, augment=True)
note(bool(clip[0]["flip"].all()), "the flip reached every keyframe of the clip")
bad = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is not None and not torch.isfinite(p.grad).all()]
none = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
note(not bad, f"gradients finite ({len(bad)} non-finite)")
print(f"        trainable parameters without a gradient in this clip: {len(none)} {none[:8]}", flush=True)

# 3 dense outputs and the post-processing
model.eval()
model.cfg.dense_outputs = True
bd = T.to_device(collate_cached([val[0], val[57]]), "cuda")
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    res = model(bd, dts=(-0.5, 0.0, 0.5))
e = check_outputs(res, (-0.5, 0.0, 0.5))
note(not e, f"check_outputs with dense outputs: {e[:4] if e else 'clean'}")
p0 = res["out"][0.0]
note(set(p0["boxes"]) >= {"w", "cx", "cy", "yaw", "vx", "vy", "hx", "hy"} and p0["lines"].shape[1] == len(model._dist_idx),
     f"dense outputs present: boxes {sorted(p0['boxes'])}, lines {tuple(p0['lines'].shape)}")
got = model.boxes(res, 0.0, k=100, threshold=0.05)
ref = decode({k: v.float().cpu() for k, v in p0["obj"].items()}, model.grid, NUSC_GROUPS, k=100, threshold=0.05)
same = [len(g) == len(r) and all(abs(a["cx"] - c["cx"]) < 1e-4 and abs(a["score"] - c["score"]) < 1e-6 for a, c in zip(g, r))
        for g, r in zip(got, ref)]
note(all(same), f"model.boxes = object_head.decode on real maps ({[len(g) for g in got]} boxes)")
model.cfg.dense_outputs = False

# 4 warm: the carried memory is read
with torch.no_grad():
    clip_d = [T.to_device(x, "cuda") for x in T.collate_clips([clips[0]], collate_cached)]
    state, prev = None, None
    for x in clip_d:
        step = T.clip_step(prev, "cuda") if prev is not None else None
        warm = model(x, dts=(0.0,), state=state, step=step)
        state, prev = warm["state"], x
    cold = model(clip_d[-1], dts=(0.0,))
d = (warm["out"][0.0]["sem"] - cold["out"][0.0]["sem"]).abs().max().item()
note(d > 1e-4, f"the carried memory changes the last keyframe (max |warm - cold| = {d:.3g})")

# 5 the precompute maps under flip
pre = Precompute.from_model(model)
bf = collate_cached([val[0], val[57]])
plain = pre(T.to_device(dict(bf), "cuda"))
flip_batch(bf, torch.tensor([True, False]))
mir = pre(T.to_device(dict(bf), "cuda"))
note(torch.equal(mir["vis_cam"][0], plain["vis_cam"][0].flip(-1)) and torch.equal(mir["vis_cam"][1], plain["vis_cam"][1]),
     "vis_cam of a flipped sample = the mirrored map")
frac = plain["vis_cam"].mean().item()
note(0.4 < frac < 0.8, f"camera map covers {frac:.1%} of the grid (the coverage note says about 62 %)")
hits = plain["radar_hits"]
note(0.0 < hits.mean().item() < 0.5, f"radar hits cover {hits.mean().item():.1%} of the grid")
print(f"dry run done ({time.time() - t0:.0f} s): {len(problems)} problem(s)" + (": " + "; ".join(problems) if problems else ""))
sys.exit(1 if problems else 0)
