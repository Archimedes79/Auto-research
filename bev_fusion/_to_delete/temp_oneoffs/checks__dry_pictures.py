import random, sys
sys.argv += ["--quick"]
sys.path.insert(0, "tests"); sys.path.insert(0, "training")
import test_flow_refine as R
import test_flow_explains as E
R.DEV = E.DEV = "cpu"
d = [R.sample(random.Random(i)) for i in range(2)]
R.picture([("S", R.Arm(True, False, 0.0, True).eval()), ("0 + P4", R.Arm(False, "P").eval())], d, "temp/checks/dry_refine_pic.png")
de = E.dataset([50_000 + i for i in range(16)], "test")[:3]
E.picture(E.Model().eval(), de, "temp/checks/dry_explains_pic.png")
print("pictures ok")
