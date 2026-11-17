import torch, json, os
for s in (0, 1):
    sd = torch.load(os.path.join(os.environ["BFLC_ROOT"], f"runs/e292_a_s{s}/model.pt"), map_location="cpu")
    a = sd["args"]
    print(s, {k: a[k] for k in a if k not in ("name",)}, sd["steps"])
    print(len(sd["state"]), sum(v.numel() for k, v in sd["state"].items() if k.startswith("pyr.")), sum(v.numel() for k, v in sd["state"].items() if k.startswith("nodes.")))
