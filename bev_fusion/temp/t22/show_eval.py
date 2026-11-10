import json
e = json.load(open("runs/t22_A1_s0/eval_metrics.json"))
m = json.load(open("runs/t22_A1_s0/metrics.json"))
print({k: v for k, v in e.items() if not k.startswith("val")})
v, v0 = e["val"], m["val"]
for h in ("16", "8"):
    for what in ("kl", "cos", "mse"):
        print(f"{what}{h}: student {v[what + h]:.3f} (metrics.json {v0[what + h]:.3f}) | " + " ".join(f"{b} {v[f'{what}{h}.{b}']:.3f}/null {v[f'null.{what}{h}.{b}']:.3f}" for b in ("near", "mid", "far")) + f" | null {v['null.' + what + h]:.3f}")
    print(f"skill{h}: {v['skill' + h]:.3f} near {v[f'skill{h}.near']:.3f} mid {v[f'skill{h}.mid']:.3f} far {v[f'skill{h}.far']:.3f}")
print("total", round(v["total"], 4), "null total", round(v["null.total"], 4), "n", v["n"])
for h in ("16", "8"):
    print(f"by_row kl{h}:", e["val_by_row"]["kl" + h])
    print(f"by_col kl{h} (min/max/mean):", min(e["val_by_col"]["kl" + h]), max(e["val_by_col"]["kl" + h]), round(sum(e["val_by_col"]["kl" + h]) / len(e["val_by_col"]["kl" + h]), 4))
