import sys, torch
a, b = torch.load(sys.argv[1]), torch.load(sys.argv[2])
bad = 0
for k in a:
    ka, kb = a[k]["keys"], b[k]["keys"]
    same_keys = ka == kb
    same_sd = same_keys and all(torch.equal(a[k]["sd"][x], b[k]["sd"][x]) for x in ka)
    same_out = a[k]["out"].keys() == b[k]["out"].keys() and all(torch.equal(a[k]["out"][x], b[k]["out"][x]) for x in a[k]["out"])
    print(f"{k:28s} keys {same_keys}  state {same_sd}  outputs {same_out}  ({len(ka)} tensors, {len(a[k]['out'])} outs)")
    bad += not (same_keys and same_sd and same_out)
print("DIFFERENT:", bad)
