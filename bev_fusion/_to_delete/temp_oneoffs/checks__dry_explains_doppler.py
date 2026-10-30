import random, sys
sys.argv += ["--quick", "--sharp", "--pyr", "--doppler"]
sys.path.insert(0, "tests"); sys.path.insert(0, "training")
import test_flow_explains as T
T.DEV = "cpu"
data = T.dataset([50_000 + i for i in range(16)], "test")[:3]
m = T.Model()
T.train(m, data, 2, random.Random(0), bs=2)
r = T.evaluate(m.eval(), data[:2])
print({k: round(v, 3) for k, v in r.items()})
T.picture(m, data, "temp/checks/dry_explains_doppler.png")
print("ok")
