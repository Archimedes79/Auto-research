"""E377 throw-away: for each textual mutant of the scratch jtoken.py, the largest deviation of the loss parts from the hand values on (NF, NF_FAR) at none weight 0.25 and 1.0 (W 0.73), and its n_near"""
import test_jtoken as TJ

JK = TJ.JK
label = 'near = near_none(batch["pos"].to(dev).float(), batch["dirv"].to(dev).float(), valid, yt)'
M = {"column of y": [(label, "near = valid & (yt == 0) & y.any(1)")], "row of y": [(label, "near = valid & (yt == 0) & y.any(-1)")], "all none": [(label, "near = valid & (yt == 0)")],
     "type only": [("_source_nll(lp, G, wn[:, None] if torch.is_tensor(wn) else wn, pos_loss)", "_source_nll(lp, G, none_weight, pos_loss)")],
     "source only": [("torch.where(yt == 0, wn, 1.0)", "torch.where(yt == 0, none_weight, 1.0)")],
     "CE denominator": [("valid, wt) for t in", "valid, None if wt is None else torch.where(yt == 0, none_weight, 1.0)) for t in")],
     "< at 2.0": [("d2 <= NEAR_R ** 2", "d2 < NEAR_R ** 2")], "no heading": [(" & (cos >= NEAR_COS)", "")],
     "no assigned": [("asg[:, None, :]", "(valid[:, None, :] & ~torch.eye(valid.shape[-1], dtype=torch.bool))")], "padding": [("return valid & (ytype == 0) & (", "return (ytype == 0) & (")],
     "ignored": [("wn = torch.where(near, none_weight_near, none_weight)", "wn = none_weight")], "CE off at 1": [("if torch.is_tensor(wn) or none_weight != 1.0 else None", "if none_weight != 1.0 else None")]}
for name, edits in M.items():
    mod = TJ._mutant(*edits)
    row = []
    for w in (0.25, 1.0):
        out, b, frames = TJ._hand((TJ.NF, TJ.NF_FAR))
        out = TJ._with_logp(out)
        wl = [[0.73 if n in TJ.NEAR_HAND else w for n in range(10)], [w] * 10]
        z, zn, t1, t3 = (x.detach().tolist() for x in (out["z"], out["zn"], out["types1"], out["types"]))
        per = [TJ._hand_loss(f, z[n], zn[n], t1[n], t3[n], wl[n]) for n, f in enumerate(frames)]
        hand = [sum(p[k] for p in per) / len(per) for k in range(4)]
        _, info = mod.jtoken_loss(out, b, none_weight=w, none_weight_near=0.73)
        dev = max(abs(info[k] - h) for k, h in zip(("nll_s", "nll_l", "ce1", "ce3"), hand))
        row.append(f"w {w}: max dev {dev:.4f}, n_near {info['n_near']}")
    print(f"{name:16s} | " + " | ".join(row))
