import random, sys
sys.path.insert(0, "tests"); sys.path.insert(0, "training")
import test_flow_refine as T
T.DEV = "cpu"
data = [T.sample(random.Random(i)) for i in range(3)]
a = T.Arm(True, True, 0.0, True, smooth=0.05)
T.train(a, data, 2, random.Random(0), bs=2)
print("S+R4+smooth", {k: round(v, 3) for k, v in T.evaluate(a.eval(), data[:2])[-1].items()})
