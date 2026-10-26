"""One-off: does the self-supervised correlation flow head (tests/test_flow_correlation.py) keep improving with more
steps? 3,000 steps, 2 seeds, same data (cached pairs), same loss and lr as the gate; held-out per-box median error at
800 / 1500 / 3000 steps, then 300 supervised fine-tuning steps from the final weights. The gate: per-box median < half
of the null baseline.
Run from the bev_fusion folder: wsl ~/miniconda3/envs/gcar/bin/python temp/checks/flow_selfsup_long.py"""
import copy
import random
import sys
import time

import torch

sys.path.insert(0, "tests")
import test_flow_correlation as T  # noqa: E402

trainset, testset = torch.load(T.CACHE, weights_only=True)
null = T.evaluate(None, testset)[2]
print(f"{len(trainset)} train / {len(testset)} held-out pairs, null per-box median {null:.2f}, gate < {0.5 * null:.2f}",
      flush=True)
CHECK = (800, 1500, 3000)
for seed in (0, 1):
    torch.manual_seed(seed)
    net = T.CorrFlow().to(T.DEV)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    rng = random.Random(seed)
    t0 = time.time()
    for it in range(1, CHECK[-1] + 1):
        b = T.batch(rng.sample(trainset, 8))
        loss = T.loss_selfsup(net, b)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if it % 250 == 0:
            print(f"  seed {seed} iter {it:4d} loss {loss.item():.4f} ({time.time() - t0:.0f} s)", flush=True)
        if it in CHECK:
            net.eval()
            ce, be, bm, n, _ = T.evaluate(net, testset)
            net.train()
            print(f"seed {seed} self-supervised {it:4d} steps | cell mean {ce:.2f} | per-box mean {be:.2f} median {bm:.2f}"
                  f" ({n} boxes) | {'PASS' if bm < 0.5 * null else 'FAIL'}", flush=True)
    ft = T.train(copy.deepcopy(net), "supervised", trainset, 300, lr=1e-3, seed=seed)
    ft.eval()
    ce, be, bm, n, _ = T.evaluate(ft, testset)
    print(f"seed {seed} + 300 supervised steps | cell mean {ce:.2f} | per-box mean {be:.2f} median {bm:.2f}", flush=True)
