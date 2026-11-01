"""t7 scratch: CPU smoke config of the real lane-branch config (2 steps, batch 1, one validation batch) -> temp/t7/smoke_<arm>.json."""
import json
import sys

arm = sys.argv[1] if len(sys.argv) > 1 else "res4"
cfg = json.load(open(f"training/configs/t7_{arm}_s0.json"))
cfg.update(name=f"smoke_{arm}", out_root="temp/t7/runs", steps=2, warmup=1, batch_size=1, workers=1, eval_every=2,
           eval_batches=1, save_every=2, log_every=1, grad_diag_every=0, device="cpu")
json.dump(cfg, open(f"temp/t7/smoke_{arm}.json", "w"), indent=1)
print(f"temp/t7/smoke_{arm}.json")
