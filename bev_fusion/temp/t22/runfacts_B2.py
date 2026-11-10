# scratch: facts of the B2 training run from log.jsonl / train.log
import json, math, re
rows = [json.loads(l) for l in open("runs/t22_B2_s0/log.jsonl") if l.strip()]
tr = [r for r in rows if "val" not in r]
bad = [r["step"] for r in tr if not all(math.isfinite(v) for v in r.values() if isinstance(v, (int, float)))]
print("logged train rows", len(tr), "non-finite rows", len(bad), bad[:5])
sp = [r["s_per_step"] for r in tr if "s_per_step" in r]
print("s/step mean %.2f max %.2f" % (sum(sp)/len(sp), max(sp)))
gn = [r["grad_norm"] for r in tr]; print("grad norm min %.1f max %.1f last %.1f" % (min(gn), max(gn), gn[-1]))
txt = open("runs/t22_B2_s0/train.log").read()
for m in re.finditer(r"(?i)(peak|max_mem|gpu mem|alloc)[^\n]*", txt):
    print(m.group(0)[:160]); 
print("resumed" in txt.lower(), "lines", txt.count("\n"))
