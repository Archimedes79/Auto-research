import random, sys
sys.path.insert(0, "tests"); sys.path.insert(0, "training")
import test_flow_refine as T
T.DEV = "cpu"
data = [T.sample(random.Random(i)) for i in range(3)]
a = T.Arm(True, True, 0.0, True, doppler=True, doppler_every=True, smooth=0.05)
T.train(a, data, 2, random.Random(0), bs=2)
r = T.evaluate(a.eval(), data[:2])
print("S+DD+R4+smooth", [{k: round(v, 2) for k, v in x.items()} for x in (r[0], r[-1])])
