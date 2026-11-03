"""Summarise runs/t1_r0/log.jsonl: validation rows and the training groups every 400 steps (review, read-only)."""
import json

rows = [json.loads(l) for l in open("runs/t1_r0/log.jsonl")]
for r in rows:
    if "val" in r:
        print("VAL", r["step"], {k: round(v, 3) for k, v in r["val"].items()})
tr = [r for r in rows if "grad_norm" in r]
for r in tr[::20]:
    print(r["step"], round(r["total"], 2), {k[6:]: round(v, 2) for k, v in r.items() if k.startswith("group.")},
          {k[2:]: round(v, 2) for k, v in r.items() if k.startswith("s.")})
for r in rows:
    if any(k.startswith("diag") or "grad_diag" in k or "conflict" in k for k in r):
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
