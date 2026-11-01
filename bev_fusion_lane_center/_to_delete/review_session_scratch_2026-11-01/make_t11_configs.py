"""throw-away: the two t11 configs = t10_dual_s0.json + the card's changes"""
import json
src = json.load(open("training/configs/t10_dual_s0.json"))
for name, moving in (("t11_vel_ctl", False), ("t11_vel_hurdle", True)):
    c = dict(src, name=name, init_from="runs/t10_dual_s0/ckpt_00003000.pt", steps=1000, lr=1.5e-4, warmup=50, eval_every=500)
    c["model"] = dict(src["model"], **({"object_moving": True} if moving else {}))
    c["loss"] = dict(src["loss"], warmup_hold=0.0, warmup_ramp=0.01)
    keys = list(src) + ["init_from"]
    c = {k: c[k] for k in keys}
    with open(f"training/configs/{name}.json", "w") as f:
        json.dump(c, f, indent=1)
        f.write("\n")
