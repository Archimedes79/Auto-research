p = "temp/calib/probe_model_stages.py"
s = open(p, encoding="utf-8").read()
s = s.replace('''    Y = {k: [] for k in TARGETS}
    M = []
    rng = np.random.default_rng(0)''', '''    Y = {k: [] for k in TARGETS}
    M = []
    O = {k: [] for k in TARGETS}                                  # the model's own output logit (semantic head) of each target
    spec_idx = ev.layer_index(model.spec)''')
s = s.replace('''                for k in TARGETS:
                    Y[k].append(torch.from_numpy((np.asarray(g[k], np.float32) > 0.5).reshape(-1)[flat]))''', '''                for k in TARGETS:
                    Y[k].append(torch.from_numpy((np.asarray(g[k], np.float32) > 0.5).reshape(-1)[flat]))
                    O[k].append(res["out"][0.0]["sem"][b, spec_idx[k][0]].reshape(-1)[torch.from_numpy(flat)].float())''')
s = s.replace("    return {s: torch.cat(v) for s, v in feats.items() if v}, {k: torch.cat(v) for k, v in Y.items()}, torch.cat(M)",
              "    return {s: torch.cat(v) for s, v in feats.items() if v}, {k: torch.cat(v) for k, v in Y.items()}, torch.cat(M), {k: torch.cat(v) for k, v in O.items()}")
s = s.replace("Ft, Yt, Mt = collect(train_ds, n_train)", "Ft, Yt, Mt, Ot = collect(train_ds, n_train)")
s = s.replace("Fv, Yv, Mv = collect(val_ds, n_val)", "Fv, Yv, Mv, Ov = collect(val_ds, n_val)")
s = s.replace('''print("\n| target | " + " | ".join(f"probe {s}" for s in Ft) + " | positives (val) |")
print("|---|" + "---|" * (len(Ft) + 1))''', '''print("\n| target | " + " | ".join(f"probe {s}" for s in Ft) + " | model's own output | positives (val) |")
print("|---|" + "---|" * (len(Ft) + 2))''')
s = s.replace('''    print(f"| {k} | " + " | ".join(f"{x:.3f}" for x in r) + f" | {int(Yv[k].sum())} / {len(Yv[k])} |")''', '''    thr = best_thr(Ot[k], Yt[k])
    r.append(iou(Ov[k] > thr, Yv[k]))
    print(f"| {k} | " + " | ".join(f"{x:.3f}" for x in r) + f" | {int(Yv[k].sum())} / {len(Yv[k])} |")''')
open(p, "w", encoding="utf-8", newline="\n").write(s)
