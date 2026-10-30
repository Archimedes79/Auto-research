import random, sys, time
sys.argv += ["--quick"]
sys.path.insert(0, "tests"); sys.path.insert(0, "training")
import test_flow_explains as T
t0 = time.time()
data = [T.frames(random.Random(i)) for i in range(4)]
print("render 4 sequences", round(time.time() - t0, 1), "s")
m = T.Model().to(T.DEV)
T.train(m, data, 3, random.Random(0), bs=2)
m.eval()
r = T.evaluate(m, data[:2])
for k, x in r.items(): print(f"  {k:22s} {x:.3f}")
T.picture(m, data, "temp/checks/flow_explains_dry.png", n=2)
print("ok", T.DEV)
