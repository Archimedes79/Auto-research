"""For old-chain files that the closure calls LIVE: print one import path from an entry point (scratch)."""
import os
import sys
from collections import deque
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "e285"))
import reach2  # noqa: E402

ROOT = reach2.ROOT
targets = sys.argv[1:] or ["p16_builder", "p8_lib", "p10_net", "p18_regis", "dense_head", "chain_data", "p16_loss", "p17_gate", "p10_infer", "p16_timing"]
entries = {q: None for q in reach2.active_entries}
prev = {}
dq = deque(reach2.active_entries)
seen = set(reach2.active_entries)
while dq:
    p = dq.popleft()
    for m in reach2.imports(p):
        for q in reach2.files.get(m, []):
            if q not in seen:
                seen.add(q)
                prev[q] = p
                dq.append(q)
rel = lambda p: os.path.relpath(p, ROOT).replace("\\", "/")
for t in targets:
    for q in reach2.files.get(t, []):
        chain, c = [], q
        while c in prev:
            chain.append(rel(c))
            c = prev[c]
        chain.append(rel(c))
        print(" <- ".join(chain))
