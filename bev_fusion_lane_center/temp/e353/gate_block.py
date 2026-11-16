import os, sys, json
os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT = os.environ["BFLC_ROOT"]
sys.path.insert(0, os.path.join(ROOT, "evaluation")); sys.path.insert(0, os.path.join(ROOT, "temp", "_session_scratch_e353", "pbcopy"))
import jprune_b as PB
allrows = []
for S in (0, 1):
    rows, ok = PB.gate(S, f"e352diag_null_s{S}")
    allrows += rows
    print(f"seed {S}: gate {'PASS' if ok else 'FAIL'}")
print(PB.gate_md(allrows))
# worst ratio per key class and the largest differences
worst = sorted(allrows, key=lambda r: -abs(r[6]) / r[7])[:6]
print("closest to the bound (|diff| / tolerance):", [(r[0], r[1], r[2], r[3], round(abs(r[6]) / r[7], 2), f"{r[6]:+.1e}") for r in worst])
for key in PB.KEYS:
    d = [abs(r[6]) for r in allrows if r[3] == key]
    print(f"{key:12s} max |diff| over 2 seeds x 3 read-outs x (base, head): {max(d):.1e}   tolerance for the stored value {max(r[4] for r in allrows if r[3] == key):.3f}: {PB.tolerance(key, max(r[4] for r in allrows if r[3] == key)):.1e}")
