import os, sys, pickle, json
ROOT = os.environ["BFLC_ROOT"]
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import jprune_b as PB
ref = pickle.load(open(os.path.join(ROOT, "runs/e345_p000_e324_s_s0/ends_val.pkl"), "rb"))[:24]
new = pickle.load(open(os.path.join(ROOT, "runs/e352_null_s0/ends_val.pkl"), "rb"))[:24]
print(json.dumps(PB.compare_frames(ref, new), indent=1))
