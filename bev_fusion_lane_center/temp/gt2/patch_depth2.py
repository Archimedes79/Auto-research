p = "temp/e293/depth_probe.py"
s = open(p, encoding="utf-8").read()
a = s.index("        jg = J.junction_gt(g)                                                                   # control")
b = s.index("        for r in runs:\n            q = r[\"pts\"]")
new = '''        et = QE.end_table(g)                                                                    # control: the geometrically possible but FALSE pairs (curves between a true entry and a wrong exit): does the heat tell them from the true connectors?
        if et is not None:
            for (a_, b_), yy in zip(et["cand"], et["y"]):
                if yy:
                    continue
                cv = QE.curve(et["ent"][a_][:2], et["ent"][a_][2:], et["ext"][b_][:2], et["ext"][b_][2:])
                sg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(cv, axis=0), axis=1))]
                ij = np.clip(np.floor((cv + 50.0) / 1.0).astype(int), 0, 99)
                hv = cen[i][ij[:, 0], ij[:, 1]]
                dpf = tree.query(cv)[0] if tree is not None else np.full(len(cv), 99.0)
                for d_, h_, p_ in zip(sg, hv, dpf):
                    k = int(np.searchsorted(BINS, d_, side="right") - 1)
                    c = ctrl[k]
                    c["n"] += 1; c["heat"] += int(h_ >= 0.3); c["heat_sum"] += float(h_); c["paint"] += int(p_ <= 1.0)
'''
s = s[:a] + new + s[b:]
s = s.replace("ctrl = dict(n=0, heat=0, paint=0, heat_sum=0.0)\n", "ctrl = {k: dict(n=0, heat=0, paint=0, heat_sum=0.0) for k in range(len(BINS) - 1)}\n", 1)
a2 = s.index("res = dict(seed=seed")
b2 = s.index("print(json.dumps(res")
s = s[:a2] + '''res = dict(seed=seed, frames=n, runs=runs_n, seconds=round(time.time() - t0),
           false_pair_control_by_depth_m={f"{BINS[k]:g}-{BINS[k + 1] if BINS[k + 1] < 1e8 else 'inf'}": dict(points=ctrl[k]["n"], heat_ge_0_3=ctrl[k]["heat"] / max(ctrl[k]["n"], 1), heat_mean=ctrl[k]["heat_sum"] / max(ctrl[k]["n"], 1),
                                                                                                           paint_within_1m=ctrl[k]["paint"] / max(ctrl[k]["n"], 1)) for k in ctrl},
           by_depth_m={f"{BINS[k]:g}-{BINS[k + 1] if BINS[k + 1] < 1e8 else 'inf'}": summ(acc[k]) for k in acc}, by_fraction={f"{FR[k]:g}-{min(FR[k + 1], 1):g}": summ(accf[k], False) for k in accf})
''' + s[b2:]
open(p, "w", encoding="utf-8").write(s)
print("ok")
