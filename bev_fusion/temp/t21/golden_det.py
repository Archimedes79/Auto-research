import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import golden
ref = torch.load(golden.FILE, weights_only=False)
got = golden.run({})
bad = [k for k in ref if (ref[k] != got[k] if isinstance(ref[k], list) else not torch.equal(ref[k], got[k]))]
print("deterministic:", not bad, bad[:3])
