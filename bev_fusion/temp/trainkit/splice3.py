"""scratch: stage A: train() uses trainkit.checkpoint.resume and trainkit.control"""
import sys
path = sys.argv[1]
src = open(path, encoding="utf-8", newline="").read()
a = src.index("    step, damaged, was_slim = 0, [], False\n")
b = src.index("    supersede_log(out_dir, step)                                # also after a crash")
new = '''    step, was_slim = 0, False
    resumed = tk_ckpt.resume(out_dir, model, loss, opt, ema, scaler, cfg, external=_external(model), free=RESUME_FREE, switches=RESUME_SWITCHES, override_env="BF_RESUME_ANY")
    if resumed is not None:                                     # the newest READABLE checkpoint: a damaged one falls back to the one before
        step, was_slim = resumed
    elif cfg.init_from:                                         # a new run warm-started from another run's checkpoint
        fresh = warm_start(cfg.init_from, model, loss, reset=cfg.init_reset)
        lost = sorted(set(fresh) & set(frozen))
        if lost:                                                # a frozen weight that the checkpoint does not have stays what it was initialised to
            print(f"WARNING: {len(lost)} frozen state key(s) are not in the warm-start checkpoint (or under init_reset) and stay at their random initialisation, e.g. {lost[:3]}", flush=True)
        ema = EMA(model, cfg.ema, skip=frozen)
        torch.manual_seed(cfg.seed)                             # the same data order and flips whatever the new branches drew
        np.random.seed(cfg.seed)
        print(f"warm start from {cfg.init_from}; fresh keys: {fresh}", flush=True)
'''
src = src[:a] + new + src[b:]
old = '''            with open(os.path.join(out_dir, "STOPPED"), "w", encoding="utf-8") as fh:
                json.dump({"step": step, "reason": reason, "t": time.time()}, fh)
'''
assert old in src
src = src.replace(old, "            control.mark_stopped(out_dir, step, reason)\n")
assert "    _clear_control(out_dir)\n" in src
src = src.replace("    _clear_control(out_dir)\n", "    control.clear_stale(out_dir)\n")
open(path, "w", encoding="utf-8", newline="").write(src)
print("ok")
