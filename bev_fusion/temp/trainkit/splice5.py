"""scratch: stage B: train() hands the loop to trainkit.loop.fit; ProbeHooks; evaluate under ema.applied; imports"""
import sys
path = sys.argv[1]
src = open(path, encoding="utf-8", newline="").read()
new_fn = open(sys.argv[2], encoding="utf-8", newline="").read()


def cut(src, a, b, new=""):
    i, j = src.index(a), src.index(b)
    assert i < j, (a, b)
    return src[:i] + new + src[j:]


src = cut(src, "def train(cfg, eval_fn=None, datasets=None):", "# ------------------------------------------------------------------------------------------------ self-test", new_fn)
old_imp = "from trainkit.runlog import supersede_log, val_logged  # noqa: E402,F401\n"
assert old_imp in src
src = src.replace(old_imp, "from trainkit.loop import Hooks, fit  # noqa: E402\nfrom trainkit.runlog import val_logged  # noqa: E402,F401\n")
old_u = "from trainkit.util import detach_tree  # noqa: E402,F401\n"
assert old_u in src
src = src.replace(old_u, "from trainkit.util import detach_tree, prepare_run_dir, resolve_device, seed_all  # noqa: E402\n")
# evaluate: the swap of the weights is the engine's
old_ev = '''    ema.apply(model)
    model.eval()
    tot, n = {}, 0
    try:
        with torch.no_grad():
            for i, item in enumerate(loader):
                if i >= n_batches:
                    break
                clip = item if isinstance(item, list) else [item]
                out, ok, _, totals = run_clip(model, loss, clip, 1.0, device, amp, cfg, train_terms=False)
                if not ok:
                    continue
                for k, v in out.items():
                    if k == "total" or k.startswith("group.") or k in LOGGED_TERMS:
                        tot[k] = tot.get(k, 0.0) + v
                if len(clip) > 1:
                    cold, _, _ = forward_loss(model, loss, clip[-1], 1.0, device, amp, cfg, train_terms=False)
                    tot["total_last_warm"] = tot.get("total_last_warm", 0.0) + totals[-1]
                    tot["total_last_cold"] = tot.get("total_last_cold", 0.0) + _f(cold["total"])
                n += 1
        res = {k: v / max(1, n) for k, v in tot.items()}
        if eval_fn is not None:
            res.update(eval_fn(model, loader))
    finally:
        ema.restore(model)
        model.train()
    return res
'''
new_ev = '''    tot, n = {}, 0
    with ema.applied(model):
        with torch.no_grad():
            for i, item in enumerate(loader):
                if i >= n_batches:
                    break
                clip = item if isinstance(item, list) else [item]
                out, ok, _, totals = run_clip(model, loss, clip, 1.0, device, amp, cfg, train_terms=False)
                if not ok:
                    continue
                for k, v in out.items():
                    if k == "total" or k.startswith("group.") or k in LOGGED_TERMS:
                        tot[k] = tot.get(k, 0.0) + v
                if len(clip) > 1:
                    cold, _, _ = forward_loss(model, loss, clip[-1], 1.0, device, amp, cfg, train_terms=False)
                    tot["total_last_warm"] = tot.get("total_last_warm", 0.0) + totals[-1]
                    tot["total_last_cold"] = tot.get("total_last_cold", 0.0) + _f(cold["total"])
                n += 1
        res = {k: v / max(1, n) for k, v in tot.items()}
        if eval_fn is not None:
            res.update(eval_fn(model, loader))
    return res
'''
assert old_ev in src
src = src.replace(old_ev, new_ev)
open(path, "w", encoding="utf-8", newline="").write(src)
print("ok")
