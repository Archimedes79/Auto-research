"""General training harness for every model of the project (new code uses it; the loop-10..17 scripts have their own copies of the same boilerplate).

    import trainer as T
    run = T.Run("e210_demo")                                   # runs/e210_demo/{log.txt, status.json, ckpt.pt, model.pt}
    T.fit(model, loss_fn, run, steps=4800, batch_fn=make_batch, lr=6e-4)

  loss_fn(model, batch) -> (loss tensor, dict of floats for the log)
  batch source: batch_fn(step) -> batch   (deterministic in `step`: a resumed run sees exactly the same batches)   or   loader (any re-iterable, e.g. a DataLoader; resume then reshuffles)
What it adds to a plain loop (each of these was missing in a run that was lost):
  * checkpoint every `ckpt_every` steps (model, optimiser, scaler, schedule, step, RNG states, loss history), written atomically; `resume=True` continues from it after a crash or a kill
  * `model.pt` + status.json + the `.done` flag only after the last step succeeded; an exception writes status "failed" with the traceback and leaves the checkpoint for the next start
  * AMP (bf16 or fp16 with GradScaler), gradient clipping, AdamW with warm-up + one-cycle / cosine schedule
  * non-finite steps (loss or gradient, also the steps the fp16 scaler skips) are skipped and counted; `max_bad` of them in a row, or a run in which no step ever updated the weights, FAIL (no .done flag, status "failed"); a skipped step still counts toward `steps`, `updates` in status.json is the number of optimizer steps made
  * guard against loss spikes: a moving average above `guard_ratio` x best triggers a rollback to the last checkpoint with fresh Adam moments and lr x 0.7 (the rule of dense_head.py)
  * periodic `eval_fn(model, step) -> dict` written to status.json
"""
import bflc_paths as BP  # project paths from the location of this checkout
import json
import math
import os
import random
import time
import traceback

import numpy as np
import torch

RUNS = BP.RUNS


class Run:
    """a run directory with a log and a status file"""

    def __init__(self, name, root=RUNS):
        self.name, self.dir = name, os.path.join(root, name)
        os.makedirs(self.dir, exist_ok=True)
        self.t0 = time.time()
        self._status = {}
        p = os.path.join(self.dir, "status.json")
        if os.path.exists(p):
            try:
                self._status = json.load(open(p))
            except Exception:
                pass

    def log(self, *a):
        line = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a)
        print(line, flush=True)
        with open(os.path.join(self.dir, "log.txt"), "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def status(self, **kw):
        self._status.update(kw)
        tmp = os.path.join(self.dir, "status.json.tmp")
        json.dump(self._status, open(tmp, "w"), indent=1, default=float)
        os.replace(tmp, os.path.join(self.dir, "status.json"))

    def path(self, name):
        return os.path.join(self.dir, name)

    def mark_done(self):
        open(self.path(".done"), "w").write(time.strftime("%Y-%m-%d %H:%M:%S\n"))


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _rng_state():
    return dict(py=random.getstate(), np=np.random.get_state(), torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def _set_rng(s):
    random.setstate(s["py"])
    np.random.set_state(s["np"])
    torch.set_rng_state(s["torch"])
    if s.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(s["cuda"])


def _lr_at(step, steps, lr, warmup, sched):
    w = max(int(warmup * steps), 1)
    if step < w:
        return lr * (step + 1) / w
    t = (step - w) / max(steps - w, 1)
    if sched == "cosine":
        return lr * 0.5 * (1 + math.cos(math.pi * t))
    if sched == "constant":
        return lr
    return lr * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * t)))  # "onecycle": warm-up then cosine to 2 % of the peak


def _endless(loader):
    """the batches of a data loader, pass after pass; every pass iterates the loader anew (itertools.cycle would replay the batches of the
    FIRST pass: no reshuffling, no new augmentation, and every batch of the epoch kept in memory)"""
    while True:
        n = 0
        for batch in loader:
            n += 1
            yield batch
        if n == 0:
            raise ValueError("the data loader gave no batch (a generator or iterator can be read once: pass something re-iterable, e.g. a DataLoader or a list)")


def fit(model, loss_fn, run, *, steps, batch_fn=None, loader=None, lr=6e-4, wd=1e-4, betas=(0.9, 0.98), eps=1e-6, warmup=0.05, sched="onecycle", amp="bf16", clip=3.0,
        ckpt_every=500, resume=True, eval_fn=None, eval_every=0, guard=True, guard_ratio=1.2, max_rollbacks=6, log_every=100, seed=0, device="cuda", args=None, params=None, max_bad=20):
    assert (batch_fn is None) != (loader is None), "give batch_fn(step) or loader"
    seed_all(seed)
    model.to(device)
    opt = torch.optim.AdamW(params if params is not None else model.parameters(), lr=lr, weight_decay=wd, betas=betas, eps=eps)
    use_amp = amp in ("bf16", "fp16") and device == "cuda"
    dtype = torch.bfloat16 if amp == "bf16" else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and amp == "fp16")
    step, hist, n_skip, n_roll, lr_scale, best = 0, [], 0, 0, 1.0, None
    ck = run.path("ckpt.pt")
    if resume and os.path.exists(ck):
        s = torch.load(ck, map_location=device, weights_only=False)
        model.load_state_dict(s["model"])
        opt.load_state_dict(s["opt"])
        scaler.load_state_dict(s["scaler"])
        step, hist, n_skip, n_roll, lr_scale, best = s["step"], s["hist"], s["n_skip"], s["n_roll"], s["lr_scale"], s["best"]
        _set_rng(s["rng"])
        run.log(f"resumed from {ck} at step {step}")
    it = None if batch_fn else _endless(loader)
    bad = 0                                                              # non-finite steps in a row

    def save_ckpt():
        tmp = ck + ".tmp"
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), scaler=scaler.state_dict(), step=step, hist=hist, n_skip=n_skip, n_roll=n_roll, lr_scale=lr_scale, best=best, rng=_rng_state()), tmp)
        os.replace(tmp, ck)

    def skipped_step(what):                                              # one non-finite step: counted, no update; too many in a row = diverged
        nonlocal n_skip, bad, step
        n_skip += 1
        bad += 1
        opt.zero_grad(set_to_none=True)
        step += 1
        run.log(f"step {step}: non-finite {what} skipped ({n_skip} so far, {bad} in a row)")
        if bad >= max_bad:
            raise RuntimeError(f"{bad} non-finite steps in a row: the run has diverged")

    run.status(state="running", steps=steps, args=args, started=time.strftime("%Y-%m-%d %H:%M:%S"))
    t0 = time.time()
    model.train()
    try:
        while step < steps:
            batch = batch_fn(step) if batch_fn else next(it)
            for g in opt.param_groups:
                g["lr"] = _lr_at(step, steps, lr, warmup, sched) * lr_scale
            with torch.autocast(device_type="cuda", dtype=dtype, enabled=use_amp):
                loss, info = loss_fn(model, batch)
            updated = False
            if not torch.isfinite(loss):
                skipped_step("loss")
            else:
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)                                     # (a no-op without the fp16 scaler)
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), clip if clip else float("inf"))     # (inf: only measures)
                if not scaler.is_enabled() and not torch.isfinite(gn):   # a finite loss can have a non-finite gradient (sqrt at 0): no update
                    skipped_step("gradient")
                else:
                    scale0 = scaler.get_scale()                          # (1.0 without the fp16 scaler)
                    scaler.step(opt)
                    scaler.update()
                    if scaler.get_scale() < scale0:                      # fp16: the scaler found inf / nan gradients, skipped the update, lowered its scale
                        skipped_step("gradient")
                    else:
                        bad = 0
                        step += 1
                        hist.append(float(loss))
                        updated = True
            if updated and step % log_every == 0:
                cur = float(np.mean(hist[-log_every:]))
                run.log(f"step {step}/{steps} loss {cur:.4f} lr {opt.param_groups[0]['lr']:.2e} {(time.time() - t0) / max(step, 1):.3f} s/step " + " ".join(f"{k} {v:.3f}" for k, v in (info or {}).items()))
                if guard and step >= 4 * log_every and os.path.exists(ck):
                    best = cur if best is None else min(best, cur)
                    if cur > guard_ratio * best and n_roll < max_rollbacks:
                        s = torch.load(ck, map_location=device, weights_only=False)
                        model.load_state_dict(s["model"])
                        opt.state.clear()
                        lr_scale *= 0.7
                        n_roll += 1
                        bad = 0
                        step = s["step"]
                        run.log(f"ROLLBACK to step {step}: loss {cur:.3f} > {guard_ratio} x best {best:.3f}; fresh Adam moments, lr scale {lr_scale:.2f}")
                        continue
            if ckpt_every and step % ckpt_every == 0:
                save_ckpt()
            if eval_fn and eval_every and step % eval_every == 0:
                model.eval()
                with torch.no_grad():
                    res = eval_fn(model, step)
                model.train()
                run.status(**{f"eval_{step}": res})
                run.log("eval", step, res)
        if steps > 0 and not hist:                                       # every step skipped: no weight was ever updated, that is no finished run
            raise RuntimeError(f"no optimizer update in {steps} steps ({n_skip} skipped as non-finite)")
        share = n_skip / max(steps, 1)                                   # isolated skips never trip max_bad: a run that skipped a lot says so
        if share > 0.05:
            run.log(f"WARNING: {n_skip} of {steps} steps ({share:.0%}) were skipped as non-finite; the run is finished but look at its curve")
        save_ckpt()
        torch.save(dict(state=model.state_dict(), args=args, steps=steps), run.path("model.pt"))
        run.status(state="done", step=step, seconds=round(time.time() - t0, 1), final_loss=float(np.mean(hist[-log_every:])) if hist else None, skipped=n_skip, rollbacks=n_roll,
                   updates=len(hist), skipped_share=round(share, 4))
        run.mark_done()
        run.log("done", step)
    except BaseException as e:  # keep the checkpoint, tell the next start what happened
        run.status(state="failed", step=step, error=repr(e), trace=traceback.format_exc()[-1500:])
        run.log("FAILED at step", step, repr(e))
        raise
    return hist
