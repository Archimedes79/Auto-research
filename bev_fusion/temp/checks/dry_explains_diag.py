import random, sys
sys.argv += ["--quick", "--sharp", "--pyr", "--doppler", "--rawwarp", "1.0"]
sys.path.insert(0, "tests"); sys.path.insert(0, "training")
import test_flow_explains as T
T.DEV = "cpu"
data = T.dataset([50_000 + i for i in range(16)], "test")[:3]
m = T.Model()
r = T.evaluate(m.eval(), data[:2])
print({k: round(v, 3) for k, v in r.items() if k.startswith("EPE")})
