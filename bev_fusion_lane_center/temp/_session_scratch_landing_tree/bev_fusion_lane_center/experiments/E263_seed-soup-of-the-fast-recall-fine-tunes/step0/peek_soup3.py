"""read-only: weight distances inside the f1 lineage (rc1, rc1s1, rc2, rc4, f1eq) and the j1 lineage (j1rc6), relative to their starts"""
import sys, json, torch
torch.set_num_threads(2)
root = sys.argv[1]
runs = ["e123_D7_r3_f1", "e123_D7_r3_rc1", "e18_D7_rc1s1", "e18_D7_rc2", "e18_D7_rc4", "e18_D7_f1eq", "e123_D7_r3_j1", "e18_D7_j1rc6", "e18_D7_j1rc"]
sd = {r: torch.load(f"{root}/runs/e123/{r}/model.pt", map_location="cpu") for r in runs}
F = {r: torch.cat([v.float().flatten() for k, v in sd[r].items() if v.dtype.is_floating_point and "running" not in k]) for r in runs}
rel = lambda a, b: round(float((F[a] - F[b]).norm() / F[b].norm()), 4)
for r in runs[1:6]:
    print(r, "from f1", rel(r, "e123_D7_r3_f1"), "from rc1", rel(r, "e123_D7_r3_rc1"))
print("j1rc6 from j1", rel("e18_D7_j1rc6", "e123_D7_r3_j1"), "from j1rc", rel("e18_D7_j1rc6", "e18_D7_j1rc"))
for r in ["e123_D7_r3_rc1", "e18_D7_rc2", "e18_D7_j1rc"]:
    c = json.load(open(f"{root}/runs/e123_ladder/cards/{r}.json", encoding="utf-8-sig"))
    print(r, "init", c.get("init"), "w", c["args"].get("w"), "seed", c.get("seed"), "u", c.get("u"), "lr", c["args"].get("lr"), "stem_lr", c["args"].get("stem_lr"), "train_s", round(c.get("train_seconds", 0)))
