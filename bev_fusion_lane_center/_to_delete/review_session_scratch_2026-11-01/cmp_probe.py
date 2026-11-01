import torch
a, b = (torch.load(f"/tmp/t11probe/{n}.pt", weights_only=False) for n in ("old", "new"))
for part in ("model", "loss", "ema"):
    assert a[part].keys() == b[part].keys(), part
    diff = [k for k in a[part] if not torch.equal(a[part][k], b[part][k])]
    assert not diff, (part, diff[:5])
    print(part, len(a[part]), "tensors bit-identical")
for ra, rb in zip(a["rows"], b["rows"]):
    common = set(ra) & set(rb) - {"t", "s_per_step"}
    sa = {k: ra[k] for k in common} if "val" not in ra else ra["val"]
    sb = {k: rb[k] for k in common} if "val" not in rb else {k: v for k, v in rb["val"].items() if k in ra["val"]}
    assert sa == sb, (sa, sb)
    print("row", ra.get("step"), "equal on", len(sa), "keys; new keys:", sorted(set(rb.get("val", rb)) - set(ra.get("val", ra))))
print(sorted(b["rows"][0]))
print(sorted(b["rows"][-1]["val"]))
