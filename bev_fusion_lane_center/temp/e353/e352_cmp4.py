import os, sys, pickle, json
ROOT = os.environ["BFLC_ROOT"]
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import jprune_b as PB
ref = pickle.load(open(os.path.join(ROOT, "temp/e276/predfull_e273_p_s0.pkl"), "rb"))[:24]
new = pickle.load(open(os.path.join(ROOT, "temp/e352/check_s0/predfull_null_s0.pkl"), "rb"))
print(json.dumps(PB.compare_dumps(ref, new), indent=1))
