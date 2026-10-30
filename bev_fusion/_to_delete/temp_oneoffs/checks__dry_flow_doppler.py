import random, sys
sys.path.insert(0, "tests"); sys.path.insert(0, "training")
import test_flow_refine as T
T.DEV = "cpu"
data = [T.sample(random.Random(i)) for i in range(3)]
for name, refine in (("S + D", False), ("S + D + P4", "P")):
    a = T.Arm(True, refine, 0.0, True, doppler=True)
    T.train(a, data, 2, random.Random(0), bs=2)
    r = T.evaluate(a.eval(), data[:2])
    print(name, {k: round(v, 3) for k, v in r[0].items()}, "last", {k: round(v, 3) for k, v in r[-1].items()})
