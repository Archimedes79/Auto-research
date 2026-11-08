p = "bev_fusion/temp/t16/probe_modules.py"
s = open(p, encoding="utf-8").read()
a = """        modules.setdefault("features(S)", []).append(stat(res["features"]))
"""
b = """        modules.setdefault("features(S)", []).append(stat(res["features"]))
        # static-cell check (Temporal Fusion chat's question): is the field on static cells zero-mean noise or an offset along the ego velocity?
        gv = torch.as_tensor(np.asarray(gt_cur["velocity"])).float()
        und = torch.as_tensor(np.asarray(gt_cur["undefined"])).bool()
        static = (gv.norm(dim=0) < 0.3) & ~und
        movg = (gv.norm(dim=0) >= 1.0) & ~und
        for lvl, v in [("v_hat", out0["v_hat"])] + [(f"v_levels.{kk}", vv) for kk, vv in (out0.get("v_levels") or {}).items()]:
            v = v[0].float().cpu()
            vs = v[:, static]
            rows.append({"frame": ref.key, "level": lvl, "ego_v_mps": float(batch["v"][0]), "static_cells": int(static.sum()),
                         "static_mean_vx": float(vs[0].mean()), "static_mean_vy": float(vs[1].mean()), "static_std_x": float(vs[0].std()),
                         "static_std_y": float(vs[1].std()), "static_epe": float(vs.norm(dim=0).mean()),
                         "moving_cells": int(movg.sum()), "moving_epe": float((v[:, movg] - gv[:, movg]).norm(dim=0).mean()) if movg.any() else None,
                         "moving_gt_speed": float(gv[:, movg].norm(dim=0).mean()) if movg.any() else None})
"""
assert s.count(a) == 1
s = s.replace(a, b)
a = """        gt = sample.reference[0.0]
"""
assert s.count(a) == 1
s = s.replace(a, a)                                   # gt is defined later (for the images); the static check needs it earlier
a2 = """        base = {n: {p: t.clone() for p, t in d.items()} for n, d in store.items()}
"""
b2 = """        base = {n: {p: t.clone() for p, t in d.items()} for n, d in store.items()}
        gt_cur = sample.reference[0.0]
"""
assert s.count(a2) == 1
s = s.replace(a2, b2)
a3 = """    print("\nMODULE OUTPUTS (mean over frames):")"""
b3 = """    print("\nSTATIC / MOVING CELLS of the velocity field (per frame and level; ego speed, mean field on static cells, EPE):")
    for r in rows:
        print("  ", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k != "frame"})
    json.dump(rows, open(os.path.join(out_dir, "static_check.json"), "w"), indent=1)
    print("\nMODULE OUTPUTS (mean over frames):")"""
assert s.count(a3) == 1
s = s.replace(a3, b3)
open(p, "w", encoding="utf-8").write(s)
print("patched")
