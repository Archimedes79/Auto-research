p = "training/grid/qpredict.py"
s = open(p, encoding="utf-8").read()
s = s.replace("def predict_from_out(out, batch, keep_maps=False):", "def predict_from_out(out, batch, keep_maps=False, min_p=0.02):")
s = s.replace("        sel = (L[\"pb\"] == b) & v[L[\"pa\"]] & v[L[\"pc\"]]", "        sel = (L[\"pb\"] == b) & v[L[\"pa\"]] & v[L[\"pc\"]] & (L[\"logit\"].float().sigmoid() >= min_p)          # links below min_p are dropped (memory); the read-out threshold is 0.5")
s = s.replace("def predict(model, batch, keep_maps=False):", "def predict(model, batch, keep_maps=False, min_p=0.02):")
s = s.replace("    return predict_from_out(out, batch, keep_maps=keep_maps)", "    return predict_from_out(out, batch, keep_maps=keep_maps, min_p=min_p)")
open(p, "w", encoding="utf-8").write(s)
