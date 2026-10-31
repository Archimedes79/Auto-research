import time, numpy as np, autogt

t0 = time.time()
gen = autogt.AutoGT("~/data/nuscenes", "v1.0-mini")
t1 = time.time()
toks = gen.keyframes("scene-0061")
t2 = time.time()
g = gen.keyframe(toks[8])
t3 = time.time()
print("init s", round(t1 - t0, 2), "keyframes s", round(t2 - t1, 2), "keyframe() s", round(t3 - t2, 2), "total s", round(t3 - t0, 2))
print("n keyframes", len(toks), "token", toks[8])
layers = {k: v for k, v in g.items() if k != "meta"}
print("n layers", len(layers), "shapes", sorted({tuple(np.shape(v)) for v in layers.values()}))
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
print("signature gen", gen.signature(), "current", autogt.current_signature(), "equal", gen.signature() == autogt.current_signature())
