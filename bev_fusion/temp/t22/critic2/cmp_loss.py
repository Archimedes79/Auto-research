import sys, torch
a, b = torch.load(sys.argv[1], weights_only=False), torch.load(sys.argv[2], weights_only=False)
for k in a:
    if k.startswith("fl_") and not k.endswith("keys"):
        print(k, "keys equal:", a[k + "_keys"] == b[k + "_keys"], "res keys equal:", a[k + "_reskeys"] == b[k + "_reskeys"], "values bit equal:", set(a[k]) == set(b[k]) and all((torch.equal(a[k][n], b[k][n]) if torch.is_tensor(a[k][n]) else a[k][n] == b[k][n]) for n in a[k]))
print("run_clip rows equal:", a["clip"] == b["clip"], [n for n, _ in a["clip"][0]][:14])
