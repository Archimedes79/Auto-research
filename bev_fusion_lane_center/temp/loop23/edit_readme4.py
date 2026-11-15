p = "evaluation/README.md"
L = open(p, encoding="utf-8").read().split("\n")
i = [k for k, ln in enumerate(L) if ln.startswith("| `golden.json` |")]
assert len(i) == 1
L.insert(i[0] + 1, "| `lane_holdout_scenes.json` | the scene names of the held-out sets: VAL (22 scenes) and TEST (21), for other projects whose splits overlap them (the BEV system's 82 validation scenes contain all 43; D47 (d)) |")
open(p, "w", encoding="utf-8").write("\n".join(L))
