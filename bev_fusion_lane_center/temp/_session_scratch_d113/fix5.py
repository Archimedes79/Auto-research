import io
p = "training/README.md"
s = io.open(p, encoding="utf-8").read()
a = "`build` (GPU, resumable, through `run.py`), \"\n                   \"`fixture`"
a = "`build` (GPU, resumable, through `run.py`), `fixture` (the cache against B's stored dump, bit for bit), `report`, `selftest`"
b = "`build` (GPU, resumable, through `run.py`), `fixture` (the cache against B's stored dump, bit for bit), `same` (two builds equal in every field), `report`, `summary` (the text record `experiments/E365_e2e-token-champion-s1/cache_manifest.json`), `selftest`"
assert s.count(a) == 1, s.count(a)
io.open(p, "w", encoding="utf-8").write(s.replace(a, b))
