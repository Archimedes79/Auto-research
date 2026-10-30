import random, sys
sys.path.insert(0, "tests"); sys.path.insert(0, "training")
import test_flow_refine as T
T.DEV = "cpu"
data = [T.sample(random.Random(i)) for i in range(3)]
for name, hyp, refine, sharp in (("S", True, False, True), ("S + R4", True, True, True), ("0 + P4", False, "P", False), ("S + P4", True, "P", True)):
    a = T.Arm(hyp, refine, 0.0, sharp)
    T.train(a, data, 2, random.Random(0), bs=2)
    r = T.evaluate(a.eval(), data[:2])
    print(name, len(r), "stages", {k: round(v, 3) for k, v in r[-1].items()})
