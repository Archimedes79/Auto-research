import sys
sys.path.insert(0, "training")
import distill_student as D
cfg = D.DistillConfig()
st = D.FeatureStore(cfg.feats16)
for name in ("small", "large"):
    lists = D.splits.load(name, "splits")
    n = {p: len(D.split_tokens(st, lists[p], cfg.img_root)) for p in ("train", "val")}
    print(name, len(lists["train"]), len(lists["val"]), n, "steps/epoch at batch 16:", n["train"] // 16)
