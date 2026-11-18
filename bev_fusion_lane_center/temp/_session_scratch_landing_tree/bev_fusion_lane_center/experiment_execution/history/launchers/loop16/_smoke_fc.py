"""smoke test of the fork-coverage loss on a few batches (j1 weights, beam 288 / 144); prints the loss parts"""
import json
import subprocess
import sys
sys.path.insert(0, ".")
import p16_triage as T

CARRY = ["reach", "spf2", "junc", "temporal", "stage_b", "stage_c", "b", "mcl", "ver_mode", "link_mode", "pi_mode", "detach_pos", "dedup", "ctx", "pattern", "seeds", "warm", "revisit", "constr", "D", "M", "K0", "seed_gscale", "tolerant", "local_r"]
pc = T.card("e123_D7_r3_j1")["cfg"]
sets = [f"{k}={json.dumps(pc[k])}" for k in CARRY if k in pc and k not in ("M", "K0")] + ["M=288", "K0=144"]
cmd = ["python", "p16_train.py", "--design", "D7", "--name", "e123_smoke_fc", "--u", "1", "--smoke", "--init", "e123_D7_r3_j1", "--eval", "none", "--workers", "2", "--lr", "5e-4", "--w", "fc=1.0", "--set"] + sets
print(" ".join(cmd[:14]), "...")
r = subprocess.run(cmd, capture_output=True, text=True)
print(r.stdout[-1800:])
print(r.stderr[-1500:])
