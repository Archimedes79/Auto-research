import sys, torch
a, b = torch.load(sys.argv[1], weights_only=False), torch.load(sys.argv[2], weights_only=False)
for k in a:
    x, y = a[k], b[k]
    ok = {"keys": x["keys"] == y["keys"], "pnames": x["pnames"] == y["pnames"], "res_keys": x["res_keys"] == y["res_keys"],
          "init": all(torch.equal(x["sd"][n], y["sd"][n]) for n in x["sd"]) and set(x["sd"]) == set(y["sd"]),
          "outs": set(x["outs"]) == set(y["outs"]) and all(torch.equal(x["outs"][n], y["outs"][n]) for n in x["outs"]),
          "feat": torch.equal(x["feat"], y["feat"]),
          "grads": set(x["grads"]) == set(y["grads"]) and all(torch.equal(x["grads"][n], y["grads"][n]) for n in x["grads"])}
    print(k, ok)
