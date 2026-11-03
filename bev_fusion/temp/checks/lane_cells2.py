import random, sys, torch
sys.argv += ["--quick"]
sys.path.insert(0, "tests"); sys.path.insert(0, "training")
import test_flow_explains as T
maps, vgt, bid, steps = T.frames(random.Random(50_000))
for f in range(3):
    lane = maps[f, 0]
    print(f"frame {f}: lane cells > 0.05: {int((lane > 0.05).sum())}, vehicle cells > 0.5: {int((maps[f, 1] > 0.5).sum())}")
