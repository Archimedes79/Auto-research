import json, re
reg = json.load(open("models/registry.json"))
src = open("training/pipeline/p16_infer.py").read()
sig = re.search(r"def assemble\((.*?)\):\n", src, re.S).group(1)
names = {p.split("=")[0].strip() for p in sig.split(",")}
bad = [(n, k) for n, e in reg.items() if isinstance(e, dict) for k in (e.get("asm") or {}) if k not in names]
print("assemble() parameters:", len(names), "| registry asm keys not accepted:", bad)
