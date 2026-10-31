import time, numpy as np, autogt

t0 = time.time()
tok = "017bd0fd51554fdea04b1d0b8f5d4b84"
g = autogt.load(tok)
print("load seconds", round(time.time() - t0, 3))
layers = {k: v for k, v in g.items() if k != "meta"}
print("n layers", len(layers))
shapes = sorted({tuple(np.shape(v)) for v in layers.values()})
print("shapes", shapes)
print("example shapes", {k: np.shape(layers[k]) for k in ("vehicle", "velocity", "instance")})
boxes = g["meta"]["boxes"]
print("n boxes", len(boxes))
cars = [b for b in boxes if b["group"] == "vehicle" and b["observed"] and not b["stationary"]]
print("observed moving vehicles (recipe: not stationary)", len(cars))
cars1 = [b for b in boxes if b["group"] == "vehicle" and b["observed"] and np.hypot(*b["vel"]) > 1.0]
print("observed vehicles with |vel|>1 m/s", len(cars1))
sp = [float(np.hypot(*b["vel"])) for b in boxes]
print("max speed m/s", max(sp), "km/h", max(sp) * 3.6)
sup = (g["vehicle"] > 0) & (g["flow_weight"] > 0) & (g["ignore"] == 0)
print("supervised vehicle cells", int(sup.sum()), "of", sup.size, "share", sup.mean())
print("signature", autogt.current_signature())
