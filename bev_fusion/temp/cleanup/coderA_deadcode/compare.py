"""scratch: compare safety_old.pt and safety_new.pt (written by safety.py) and print one table row per config"""
import os
import sys

import torch

here = os.path.dirname(os.path.abspath(__file__))
ORDER = ["t11_vel_hurdle_acc2", "t10_dual_s0", "t10_dual_s1", "t8_ctl_s0", "t8_ctl_s1", "t16_full.json", "t20_base.json", "t16_full_s0", "t20_base_s0"]


def load(which):
    out = {}
    for lab in ORDER:
        p = os.path.join(here, f"res_{which}", f"{lab}.pt")
        if os.path.exists(p):
            out.update(torch.load(p, weights_only=False))
    return out


old, new = load("old"), load("new")
assert set(old) == set(new), (sorted(old), sorted(new))


def maxdiff(a, b):
    if set(a) != set(b):
        return float("inf")
    return max((float((a[k] - b[k]).abs().max()) if a[k].numel() else 0.0 for k in a), default=0.0)


def rel(a, b):
    if set(a) != set(b):
        return float("inf")
    return max((abs((a[k] or 0.0) - (b[k] or 0.0)) / max(1.0, abs(a[k] or 0.0)) for k in a), default=0.0)


cols = ["config", "tensors", "keys+shapes", "order", "d out", "d seq out", "d seq state", "d loss", "d seq loss", "d grad", "d seq grad", "ckpt miss/unexp", "d ckpt out", "d ckpt seq out+state"]
print(" | ".join(cols))
bad = []
for lab in old:
    o, n = old[lab], new[lab]
    same_rows = o["rows"] == n["rows"] and o["loss_rows"] == n["loss_rows"] and sorted(o["rows"]) == sorted(n["rows"])
    same_order = o["param_order"] == n["param_order"]
    d = {"out": maxdiff(o["eval_out"], n["eval_out"]), "seq_out": maxdiff(o["seq_out"], n["seq_out"]), "seq_state": maxdiff(o["seq_state"], n["seq_state"]),
         "loss": rel(o["loss"], n["loss"]), "seq_loss": rel(o["seq_loss"], n["seq_loss"]), "grad": rel(o["grad_norm"], n["grad_norm"]), "seq_grad": rel(o["seq_grad_norm"], n["seq_grad_norm"])}
    ck = d_ck = d_cks = ""
    if "ckpt_out" in o:
        ck = f"{len(o['ckpt_missing'])}/{len(o['ckpt_unexpected'])} vs {len(n['ckpt_missing'])}/{len(n['ckpt_unexpected'])}"
        if o["ckpt_missing"] != n["ckpt_missing"] or o["ckpt_unexpected"] != n["ckpt_unexpected"] or o["ckpt_loss_missing"] != n["ckpt_loss_missing"] or o["ckpt_loss_unexpected"] != n["ckpt_loss_unexpected"]:
            bad.append((lab, "ckpt keys differ"))
        dco, dcs = maxdiff(o["ckpt_out"], n["ckpt_out"]), max(maxdiff(o["ckpt_seq_out"], n["ckpt_seq_out"]), maxdiff(o["ckpt_seq_state"], n["ckpt_seq_state"]))
        d_ck, d_cks = f"{dco:.1e}", f"{dcs:.1e}"
        if dco > 1e-6 or dcs > 1e-6:
            bad.append((lab, "ckpt outputs"))
    print(" | ".join([lab, str(len(n["rows"])), "equal" if same_rows else "DIFF", "equal" if same_order else "DIFF"] + [f"{v:.1e}" for v in d.values()] + [ck, d_ck, d_cks]))
    if not (same_rows and same_order) or max(d.values()) > 1e-6:
        bad.append((lab, "differs", d))
    # sanity: the second call really differs from the first (the memory is read), so the comparison is not vacuous
    nz = maxdiff(o["seq_out"], o["eval_out"])
    print(f"      (second call differs from the first by {nz:.2e} in the old code; state tensors compared: {len(o['seq_state'])}, outputs: {len(o['seq_out'])})")
print("ALL EQUAL" if not bad else f"PROBLEMS: {bad}")
sys.exit(1 if bad else 0)
