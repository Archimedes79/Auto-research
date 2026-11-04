"""t11 vel cause, instrumented run: train.train of the run's code snapshot with extra columns in log.jsonl, and optionally another
velocity loss - nothing in training/ is changed (the trainer looks up run_clip / forward_loss / head_loss at call time, so they are
wrapped here).

Extra columns of every logged row (means over the keyframes of the step's clip):
  diag.vel        objhead.vel@+0.0 as trained (the arm's loss)            diag.vel_l1     the L1 term of today on the same boxes
  diag.vel_zero   the L1 term of the zero predictor on the same boxes     diag.n_moving   share of supervised centres with |v| >= 0.5 m/s
  diag.act_<b>    share of (hidden unit, supervised centre) pairs with pre-activation > 0, per object-head branch b (vel, off, size, rot, hm)
  diag.alive_vel  hidden units of vel on at >= 1 supervised centre of the step     diag.gn_vel0 / gn_vel2   gradient norm of vel's two convs
Arms (config key "diag_vel_loss", read here and removed before TrainConfig): "l1" (today), "huber" (delta "diag_huber_delta", 1 m/s),
"moving" (L1 on the centres with |v| >= 0.5 m/s only: the hurdle's regression without its logit).
Run (via temp/box_vel/run_diag.sh from run.py): python train_diag.py <config.json>   (cwd = bev_fusion/, the snapshot in runs/<name>/code)
"""
import json
import os
import sys

import torch
import torch.nn.functional as F

BRANCHES = ("vel", "off", "size", "rot", "hm")
MOVING = 0.5


def install(kind="l1", delta=1.0, stop_at=0, steps=0):
    """wrap head_loss / forward_loss / run_clip of the importable chain modules (losses, object_head, train) -> the train module.
    stop_at > 0: exit 0 at the first training clip after step stop_at (its checkpoint is saved by then when save_every divides it), so
    a run with the schedule of `steps` (lr, dt warm-up) stops early; steps = the config's steps (the step is step_frac x steps)."""
    import losses as L
    import object_head as OH
    import train as T
    stash = {"rows": []}
    orig_head_loss = L.head_loss

    def head_loss(out, t, w_vel=0.2, *a, **kw):
        res = orig_head_loss(out, t, w_vel, *a, **kw)
        dev = out["hm"].device
        tt = {k: v.to(dev) for k, v in t.items()}
        m = tt["mask"].unsqueeze(-1)
        pv = OH._gather(out["vel"], tt["ind"])
        e = (pv - tt["vel"]).abs()
        res["vel_l1"] = (w_vel * (e * m).sum() / m.sum().clamp(min=1)).detach()
        res["vel_zero"] = (w_vel * (tt["vel"].abs() * m).sum() / m.sum().clamp(min=1)).detach()
        mv = m * (tt["vel"].norm(dim=-1, keepdim=True) >= MOVING).float()
        res["n_moving"] = (mv.sum() / m.sum().clamp(min=1)).detach()
        if kind == "huber":
            h = F.huber_loss(pv, tt["vel"], reduction="none", delta=delta)
            new = w_vel * (h * m).sum() / m.sum().clamp(min=1)
        elif kind == "moving":
            new = w_vel * (e * mv).sum() / mv.sum().clamp(min=1)
        else:
            new = res["vel"]
        res["total"] = res["total"] - res["vel"] + new
        res["vel"] = new
        stash["t"] = tt                                  # targets of the last object-head call (dt order: -0.5, 0, +0.5)
        stash.setdefault("t_by_call", []).append(tt)
        return res
    L.head_loss = head_loss                              # losses.object_head_terms calls it by this name

    cap = {}

    def hook(name):
        def f(_m, _i, o):
            cap.setdefault(name, []).append(o.detach())
        return f
    orig_fwd = T.forward_loss

    def forward_loss(model, loss, batch, *a, **kw):
        if not getattr(model, "_diag_hooks", False):
            head = model.obj_head.head
            for b in BRANCHES:
                getattr(head, b)[0].register_forward_hook(hook(b))
            model._diag_hooks = True
        cap.clear()
        stash["t_by_call"] = []
        out, res, gts = orig_fwd(model, loss, batch, *a, **kw)
        dts = sorted(res["out"])
        i0 = dts.index(0.0) if 0.0 in dts else None
        if i0 is not None and all(len(cap.get(b, [])) > i0 for b in BRANCHES):
            calls = stash["t_by_call"]                   # one object_head_terms call per dt of weight > 0: BEVLoss.forward goes through
            t0 = None                                    # the predictions in dt order and skips the dts whose weight is 0
            ws = [dt for dt in dts if L.dt_weight(dt, kw.get("step_frac", a[0] if a else 1.0), loss.cfg) > 0]
            if 0.0 in ws and len(calls) == len(ws):
                t0 = calls[ws.index(0.0)]
            row = {k: float(out[f"objhead.{k}@+0.0"].detach()) for k in ("vel", "vel_l1", "vel_zero", "n_moving") if f"objhead.{k}@+0.0" in out}
            if t0 is not None:
                m = t0["mask"] > 0
                for b in BRANCHES:
                    z = cap[b][i0]                       # (B, H, X, Y) pre-activation at dt 0
                    zz = z.flatten(2).gather(2, t0["ind"].unsqueeze(1).expand(-1, z.shape[1], -1))     # (B, H, M)
                    on = (zz > 0).permute(0, 2, 1)[m]    # (n_centres, H)
                    row[f"act_{b}"] = float(on.float().mean()) if len(on) else float("nan")
                    if b == "vel":
                        row["alive_vel"] = int(on.any(0).sum()) if len(on) else 0
            stash["rows"].append(row)
        return out, res, gts
    T.forward_loss = forward_loss

    orig_clip = T.run_clip

    def run_clip(model, loss, clip, *a, **kw):
        if stop_at and kw.get("backward") and a and round(float(a[0]) * steps) >= stop_at:
            print(f"train_diag: stop at step {stop_at} (schedule of {steps} steps)", flush=True)
            raise SystemExit(0)
        stash["rows"] = []
        acc, ok, dg, totals = orig_clip(model, loss, clip, *a, **kw)
        rows = stash["rows"]
        if rows:
            for k in sorted({k for r in rows for k in r}):
                vals = [r[k] for r in rows if k in r and r[k] == r[k]]
                if vals:
                    acc["diag." + k] = sum(vals) / len(vals)
        head = model.obj_head.head
        for i in (0, 2):
            g = head.vel[i].weight.grad
            acc[f"diag.gn_vel{i}"] = float(g.norm()) if g is not None else 0.0
        return acc, ok, dg, totals
    T.run_clip = run_clip
    print(f"train_diag: velocity loss {kind}" + (f" (delta {delta})" if kind == "huber" else ""), flush=True)
    return T


def main(cfg_path):
    raw = json.load(open(cfg_path))
    kind, delta = raw.pop("diag_vel_loss", "l1"), float(raw.pop("diag_huber_delta", 1.0))
    stop_at = int(raw.pop("diag_stop_at", 0))
    code = os.path.join(raw.get("out_root", "runs"), raw["name"], "code")
    sys.path.insert(0, os.path.abspath(code))
    T = install(kind, delta, stop_at, int(raw["steps"]))
    T.train(T.TrainConfig(**raw))


if __name__ == "__main__":
    main(sys.argv[1])
