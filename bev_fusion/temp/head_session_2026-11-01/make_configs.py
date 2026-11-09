"""Write the two arms of experiment t14_dontcare from t11_vel_hurdle_acc2.json (one-off; the configs are the record)."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
CFG = os.path.normpath(os.path.join(HERE, "..", "..", "training", "configs"))
base = json.load(open(os.path.join(CFG, "t11_vel_hurdle_acc2.json")))

ABOUT = ("experiment t14_dontcare, arm {arm} (../../experiments/t14_dontcare/experiment.md): t11_vel_hurdle_acc2 (the hurdle model, "
         "dual-feature chain, direction class, accumulation 2) continued for 1000 steps from its own EMA weights (init_from; optimiser and "
         "schedule fresh, as the t11 arms), seed 0{extra}; the pair differs in one field only")
for name, arm, extra, loss_extra in (
        ("t14_dc_ctl_s0", "A (control)", ", the loss exactly as in every stored run", {}),
        ("t14_dc_fusion_s0", "B", ", loss.fusion_area_only = true (D38 (b), first step): the fused heads get no loss outside the fusion "
         "area (camera x radar)", {"fusion_area_only": True})):
    out = {"_about": ABOUT.format(arm=arm, extra=extra), "name": name}
    for k, v in base.items():
        if k not in ("_about", "name", "init_from", "init_reset", "loss"):
            out[k] = v
    out["loss"] = dict(base["loss"], **loss_extra)
    out["init_from"] = "runs/t11_vel_hurdle_acc2/ckpt_00001000.pt"
    path = os.path.join(CFG, name + ".json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1)
        fh.write("\n")
    print(path, out["steps"], out["lr"], out["loss"], out["init_from"])
