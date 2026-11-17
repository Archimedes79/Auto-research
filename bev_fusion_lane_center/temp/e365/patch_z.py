p = "evaluation/README.md"
s = open(p, encoding="utf-8").read()
a = "`dump` (the head on the cache tokens, VAL or 873 TRAIN frames,"
assert s.count(a) == 1
s = s.replace(a, "`dump` (the head on the cache tokens, VAL or 873 TRAIN frames, `--every N` = every N-th frame of that sample (file names carry `_every<N>`; `score` / `sweep` take the same flag),")
open(p, "w", encoding="utf-8").write(s)
