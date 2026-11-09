"""throw-away: plant bugs in a scratch copy of training/ and check that the CPU tests of the speed work fail for each (BASELINE: unmutated, must pass).
Run from bev_fusion/: ~/miniconda3/envs/gcar/bin/python temp/speed_tools/mutate_host.py"""
import os
import shutil
import subprocess
import sys

ROOT = os.path.abspath(".")
WORK = os.path.join(ROOT, "temp", "mut_host", "bev_fusion")
PY = sys.executable
H, S, G = "test_host_overhead.py", "test_speed_flags.py", "test_grad_scaler.py"
K, R = "test_hygiene.py", "test_lr_groups.py"                          # hygiene, lr_groups

MUTANTS = [
    ("BASELINE host (no change, must PASS)", None, None, None, H),
    ("BASELINE speed (no change, must PASS)", None, None, None, S),
    ("window: last row of the window dropped", "training/object_head.py", "min(X, ix + r + 1)", "min(X, ix + r)", H),
    ("window: first column of the window dropped", "training/object_head.py", "max(0, iy - r), min(Y", "max(1, iy - r), min(Y", H),
    ("window: x and y swapped in the slice", "training/object_head.py", 'w = t["hm"][b, g, x0:x1, y0:y1]', 'w = t["hm"][b, g, y0:y1, x0:x1]', H),
    ("window: peak overwrites instead of max", "training/object_head.py", "w.copy_(torch.maximum(w, peak))", "w.copy_(peak)", H),
    ("one read: back to the read per box", "training/losses.py", "wv = float(w_host[b, ix, iy])", "wv = float(w_obj[b, 0, ix, iy])", H),
    ("one read: x and y swapped in the host index", "training/losses.py", "wv = float(w_host[b, ix, iy])", "wv = float(w_host[b, iy % 120, ix % 104])", H),
    ("one read: the weight map of sample 0 for all", "training/losses.py", "wv = float(w_host[b, ix, iy])", "wv = float(w_host[0, ix, iy])", H),
    ("EMA: the new weights get the decay (multi-tensor path)", "training/train.py", "for _, v in full], alpha=1 - d)", "for _, v in full], alpha=d)", H),
    ("EMA: the new weights get the decay (loop path)", "training/train.py", "self.shadow[k].mul_(d).add_(v.detach(), alpha=1 - d)", "self.shadow[k].mul_(d).add_(v.detach(), alpha=d)", H),
    ("EMA: frozen keys are averaged too", "training/train.py", "if k not in self.skip]\n        full", "]\n        full", H),
    ("EMA: the shadow is not decayed", "training/train.py", "torch._foreach_mul_(shadow, d)", "torch._foreach_mul_(shadow, 1.0)", H),
    ("EMA: bf16 / fp16 state through the multi-tensor call", "training/train.py", "if v.dtype in (torch.float32, torch.float64)]", "if True]", H),
    ("EMA: no guard for an empty list", "training/train.py", "        if full:  ", "        if True:  ", H),
    ("run_clip: terms narrowed to float32", "training/train.py", "tens.append(v.detach().double().reshape(()))", "tens.append(v.detach().float().reshape(()))", S),
    ("run_clip: the mean over the keyframes forgotten", "training/train.py", "acc[k] = acc.get(k, 0.0) + val / L", "acc[k] = acc.get(k, 0.0) + val", S),
    ("run_clip: the totals not collected", "training/train.py", "                totals.append(val)", "                pass", S),
    ("run_clip: a non-finite keyframe returns nothing", "training/train.py", "            acc, totals = _unpack(kf, L)\n            return acc, False, dg, totals", "            return {}, False, dg, totals", S),
    ("resume: the two switches missing in the check", "training/train.py", 'for k in ("cudnn_benchmark", "tf32", "grad_clip_split", "lr_groups") if k not in a and b.get(k)]', 'for k in ("grad_clip_split", "lr_groups") if k not in a and b.get(k)]', S),
    ("profiler: no cap on the window", "training/probe.py", "    MAX_STEPS = 8", "    MAX_STEPS = 800", S),
    ("profiler: a failing report raises", "training/probe.py", "        except Exception as e:                                  # noqa: BLE001", "        except KeyError as e:                                  # noqa: BLE001", S),
    ("profiler: a failed report leaves the profiler entered (torch aborts at exit)", "training/probe.py", "            if not self.exited:", "            if False:", S),
    ("BASELINE scaler (no change, must PASS)", None, None, None, G),
    ("scaler: gradients not unscaled before the clip", "training/train.py", "        scaler.unscale_(opt)\n    if before_clip", "        pass\n    if before_clip", G),
    ("scaler: an overflow does not lower the scale", "training/train.py", "scaler.update(scaler.get_scale() * scaler.get_backoff_factor())", "scaler.update(scaler.get_scale())", G),
    ("scaler: a norm-only overflow steps anyway", "training/train.py", "        opt.zero_grad(set_to_none=True)\n        if scaler is not None:\n            scaler.update(scaler.get_scale() * scaler.get_backoff_factor())",
     "        if scaler is not None:\n            scaler.step(opt)\n        opt.zero_grad(set_to_none=True)\n        if scaler is not None:\n            scaler.update(scaler.get_scale() * scaler.get_backoff_factor())", G),
    ("scaler: an overflow keeps the gradients", "training/train.py", "        opt.zero_grad(set_to_none=True)\n        if scaler is not None:\n            scaler.update", "        if scaler is not None:\n            scaler.update", G),
    ("bad steps: an overflow always counts", "training/train.py", "scaler.get_scale() >= 1.0:\n        return bad", "scaler.get_scale() >= 1e9:\n        return bad", G),
    ("bad steps: an overflow never counts", "training/train.py", 'if status == "ok":\n        return 0', 'if status in ("ok", "overflow"):\n        return 0', G),
    ("run_clip: the loss is not scaled", "training/train.py", "(mean_loss if scaler is None else scaler.scale(mean_loss)).backward", "(mean_loss).backward", G),
    ("amp_context: fp16 runs in bfloat16", "training/train.py", '"fp16": torch.float16}', '"fp16": torch.bfloat16}', G),
    ("make_scaler: also for bf16", "training/train.py", 'if cfg.amp != "fp16":\n        return None', 'if cfg.amp == "off":\n        return None', G),
    ("checkpoint: the scaler state is not saved", "training/train.py", 'return {"scaler": scaler.state_dict()} if scaler is not None else {}', "return {}", G),
    ("amp_context: an unknown name is accepted", "training/train.py", 'if amp != "off" and amp not in AMP_DTYPES:', "if False:", G),
    ("train: the scaler is not passed to run_clip", "training/train.py", "augment=True, scaler=scaler)", "augment=True)", G),
    ("train: the periodic checkpoint without the scaler", "training/train.py", "cfg, extra=scaler_extra(scaler))   # that dies no longer", "cfg)   # that dies no longer", G),
    ("train: the final checkpoint without the scaler", "training/train.py", ", extra=scaler_extra(scaler), slim=", ", slim=", G),
    ("train: the stop checkpoint without the scaler", "training/train.py", "cfg, extra=scaler_extra(scaler))\n            prune_checkpoints(out_dir, cfg.keep)\n            with open(os.path.join(out_dir, \"STOPPED\")", "cfg)\n            prune_checkpoints(out_dir, cfg.keep)\n            with open(os.path.join(out_dir, \"STOPPED\")", G),
    ("train: the scaler state is not restored on a resume", "training/train.py", 'scaler.load_state_dict(ck_extra["scaler"])', "pass", G),
    ("train: an overflow step counts as a good step", "training/train.py", 'skipped = status != "ok"', 'skipped = status == "nonfinite"', G),
    ("bad steps: the boundary at exactly 1", "training/train.py", "scaler.get_scale() >= 1.0:\n        return bad", "scaler.get_scale() > 1.0:\n        return bad", G),
    ("train: loss_scale not logged", "training/train.py", 'rec["loss_scale"], rec["fp16_overflows"] = scaler.get_scale(), n_overflow', "pass", G),
    ("train: overflows not counted", "training/train.py", "            n_overflow += 1\n            print(f\"step {step}: fp16 gradient overflow", "            print(f\"step {step}: fp16 gradient overflow", G),
    ("scaler: an overflow keeps the growth count", "training/train.py", '            scaler.load_state_dict({**scaler.state_dict(), "_growth_tracker": 0})', "            pass", G),
    ("train: no stop on poisoned batch-norm statistics", "training/train.py", "            if poisoned:\n                raise RuntimeError", "            if False:\n                raise RuntimeError", G),
    ("train: the diag is asked for under fp16", "training/train.py", "and scaler is None and (step + 1)", "and (step + 1)", G),
    ("make_scaler: the growth interval is dropped", "training/train.py", ", growth_interval=cfg.fp16_growth_interval)", ")", G),
    ("make_scaler: the amp name is not checked", "training/train.py", "    amp_context(cfg.amp, device)\n    if cfg.amp != \"fp16\":", "    if cfg.amp != \"fp16\":", G),
    ("BASELINE hygiene (no change, must PASS)", None, None, None, K),
    ("BASELINE lr_groups (no change, must PASS)", None, None, None, R),
    ("hygiene: the focal loss is not weighted", "training/object_head.py", "    if weight is not None:\n        w = weight.to(p.dtype)", "    if False:\n        w = weight.to(p.dtype)", K),
    ("hygiene: the object weight is not in the target mask", "training/object_head.py", 'bx.get("loss_w", 1.0), g', "1.0, g", K),
    ("hygiene: unknown velocity stays in the mov BCE", "training/object_head.py", 'mk = m * t["known"].unsqueeze(-1) if "known" in t else m', "mk = m", K),
    ("hygiene: the weight and vel_known do not travel with the box", "training/losses.py", "            if hygiene:                                                       # the weight", "            if False:                                                       # the weight", K),
    ("hygiene: the cell weights are not passed to head_loss", "training/losses.py", "cell_w=w_obj if hygiene else None", "cell_w=None", K),
    ("hygiene: the motion loss gets no weight", "training/losses.py", 'weight=sel(T["w_obj"]) if cfg.hygiene_2026_11_02 else None', "weight=None", K),
    ("hygiene: the flag does not reach the object head", "training/losses.py", "hygiene=cfg.hygiene_2026_11_02)", "hygiene=False)", K),
    ("hygiene: the motion normaliser ignores the weight", "training/timed_head.py", "/ (m * w * 2).sum().clamp(min=1)", "/ (m * 2).sum().clamp(min=1)", K),
    ("hygiene: batch_ego never raises", "training/train.py", "if strict and abs(abs(dt) - 0.5) > 1e-6:", "if False and abs(abs(dt) - 0.5) > 1e-6:", K),
    ("hygiene: forward_loss ignores the flag", "training/train.py", 'strict = bool(getattr(getattr(loss, "cfg", None), "hygiene_2026_11_02", False))', "strict = False", K),
    ("splat: half-precision geometry is not widened", "training/splat.py", "    return x.float() if x is not None and x.dtype in (torch.float16, torch.bfloat16) else x", "    return x", K),
    ("splat: splat_bev builds the cell centres in the BEV dtype", "training/splat.py", "p = grid_points(grid, bev.device, _geo_dtype(bev.dtype)).unsqueeze(0).expand(B, -1, -1)     # cell centres", "p = grid_points(grid, bev.device, bev.dtype).unsqueeze(0).expand(B, -1, -1)     # cell centres", K),
    ("splat: splat_window builds the cell centres in the BEV dtype", "training/splat.py", "p = grid_points(grid, bev.device, _geo_dtype(bev.dtype)).unsqueeze(0).expand(B, -1, -1)       # float32 cell centres", "p = grid_points(grid, bev.device, bev.dtype).unsqueeze(0).expand(B, -1, -1)       # float32 cell centres", K),
    ("lr_groups: the prefix's parameters stay in the rest", "training/train.py", "if p.requires_grad and id(p) not in claimed:", "if p.requires_grad:", R),
    ("lr_groups: the group lr is the chain's", "training/train.py", 'tag = {"base_lr": float(lr_groups[pre].get("lr", lr)), "lr_group": pre}', 'tag = {"base_lr": float(lr), "lr_group": pre}', R),
    ("lr_groups: the group warm-up is dropped", "training/train.py", '            tag["warmup"] = int(lr_groups[pre]["warmup"])', "            pass", R),
    ("lr_groups: the schedule ignores the group warm-up", "training/train.py", 'return group["base_lr"] * (f if "warmup" not in group else lr_factor(step, group["warmup"], cfg.steps, cfg.lr_min_frac))', 'return group["base_lr"] * f', R),
    ("lr_groups: the schedule ignores the group lr", "training/train.py", 'return group["base_lr"] * (f if "warmup"', 'return cfg.lr * (f if "warmup"', R),
    ("lr_groups: a resume may change them (no key check)", "training/train.py", '"grad_clip_split", "lr_groups")', '"grad_clip_split")', R),
    ("lr_groups: a typo is found only after the data are built", "training/train.py", "        lr_group_params(model, cfg.lr_groups)                   # the same", "        pass                   # the same", R),
    ("lr_groups: no lr.<prefix> column", "training/train.py", "            rec.update({f\"lr.{g['lr_group']}\": g[\"lr\"] for g in opt.param_groups if \"lr_group\" in g})", "            pass", R),
    ("lr_groups: overlap not detected", "training/train.py", "            if id(p) in seen:\n                raise ValueError(f\"lr_groups:", "            if False:\n                raise ValueError(f\"lr_groups:", R),
    ("n: the motion loss keeps the unknown-velocity cells", "training/losses.py", '        valid = valid & (T["flow_weight"][:, 0] > 0)', "        pass", K),
    ("n: the lr tags are not re-applied on a resume", "training/train.py", "        retag_lr_groups(opt, cfg)\n        if scaler is not None", "        if scaler is not None", R),
    ("n: lr_groups prefixes are not sorted", "training/train.py", "    for pre in sorted(lr_groups):", "    for pre in lr_groups:", R),
    ("n: focal positives unweighted", "training/object_head.py", "return ((pos_l * w).sum() + (neg_l * w).sum()) / (pos * w).sum().clamp(min=1)", "return (pos_l.sum() + (neg_l * w).sum()) / (pos * w).sum().clamp(min=1)", K),
    ("n: focal normaliser unweighted", "training/object_head.py", "return ((pos_l * w).sum() + (neg_l * w).sum()) / (pos * w).sum().clamp(min=1)", "return ((pos_l * w).sum() + (neg_l * w).sum()) / pos.sum().clamp(min=1)", K),
    ("n: the direction term is unweighted", "training/object_head.py", 'losses["dir"] = w_dir * (bce * mb).sum() / mb.sum().clamp(min=1)', 'losses["dir"] = w_dir * (bce * t["is_box"].unsqueeze(-1)).sum() / t["is_box"].sum().clamp(min=1)', K),
    ("n: the radius term is unweighted", "training/object_head.py", 'mc = (t["mask"] * (1 - t["is_box"])).unsqueeze(-1)', 'mc = (1 - t["is_box"]).unsqueeze(-1)', K),
    ("n: a box without vel_known counts as unknown", "training/object_head.py", 'bx.get("vel_known", True) else 0.0', 'bx.get("vel_known", False) else 0.0', K),
    ("n: strict raises only for a dt that is too large", "training/train.py", "if strict and abs(abs(dt) - 0.5) > 1e-6:", "if strict and (abs(dt) - 0.5) > 1e-6:", K),
    ("n: chain_positions does not widen half precision", "training/splat.py", "    p, v_obj, delta = _geo(p), _geo(v_obj), _geo(delta)\n    pos, vel = p, v_obj", "    pos, vel = p, v_obj", K),
    ("n: _geo also narrows float64", "training/splat.py", "x.dtype in (torch.float16, torch.bfloat16) else x\n\n\ndef _geo_dtype", "x.dtype != torch.float32 else x\n\n\ndef _geo_dtype", K),
    ("n: the correction head gets float32 centres", "training/splat.py", "corr(bev.reshape(B, C, X * Y), p.to(bev.dtype), dt, v, omega)", "corr(bev.reshape(B, C, X * Y), p, dt, v, omega)", K),
    ("profiler: item counted twice again", "training/probe.py", '("aten::item", "cudaStreamSynchronize"', '("aten::item", "aten::_local_scalar_dense", "cudaStreamSynchronize"', S),
]


if os.environ.get("ONLY"):                                                    # ONLY="n:" runs the baselines and the mutants whose name starts with it
    MUTANTS = [m for m in MUTANTS if m[1] is None or m[0].startswith(os.environ["ONLY"])]


def fresh():
    if os.path.isdir(os.path.dirname(WORK)):
        shutil.rmtree(os.path.dirname(WORK))
    os.makedirs(os.path.join(WORK, "tests"))
    shutil.copytree("training", os.path.join(WORK, "training"), ignore=shutil.ignore_patterns("__pycache__", "results", "runs"))
    for t in (H, S, G, K, R, "toy_train.py", "synthetic_world.py"):
        shutil.copy("tests/" + t, os.path.join(WORK, "tests"))
    shutil.copytree("evaluation", os.path.join(WORK, "evaluation"), ignore=shutil.ignore_patterns("__pycache__", "results", "runs", "*.png", "*.mp4"))      # bev_adapter (test_lr_groups)
    os.symlink(os.path.join(ROOT, "..", "autogt"), os.path.join(os.path.dirname(WORK), "autogt"))
    os.symlink(os.path.join(ROOT, "..", "autoresearch"), os.path.join(os.path.dirname(WORK), "autoresearch"))


survived = 0
for name, path, old, new, test in MUTANTS:
    fresh()
    if path:
        p = os.path.join(WORK, path)
        s = open(p, encoding="utf-8").read()
        assert s.count(old) == 1, (name, s.count(old))
        open(p, "w", encoding="utf-8").write(s.replace(old, new))
    r = subprocess.run([PY, "tests/" + test], cwd=WORK, capture_output=True, text=True, env=dict(os.environ, OMP_NUM_THREADS="2"))
    failed = [ln.strip() for ln in r.stdout.splitlines() if "FAIL" in ln and "GATE" not in ln]
    verdict = "PASS" if r.returncode == 0 else "FAIL"
    expect_fail = path is not None
    status = ("killed" if verdict == "FAIL" else "SURVIVED") if expect_fail else ("ok" if verdict == "PASS" else "BASELINE BROKEN")
    if status in ("SURVIVED", "BASELINE BROKEN"):
        survived += 1
    why = failed[0][:90] if failed else ((r.stdout.strip().splitlines() or [""])[-1][:60] + " / " + (r.stderr.strip().splitlines() or [""])[-1][:60] if verdict == "FAIL" else "")
    print(f"{status:16} {name}" + (f"   <- {why}" if why else ""))
print("survivors", survived)
shutil.rmtree(os.path.dirname(WORK))
