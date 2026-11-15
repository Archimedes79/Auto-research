import sys
from champion.tests.test_equivalence import run
for seed, n in ((0, 96), (1, 48)):
    worst, count = run(seed=seed, n=n)
    print("seed", seed, "frames", n, "max over all quantities", max(worst.values()), "nonzero:", {k: v for k, v in worst.items() if v}, "pieces:", count, flush=True)
