"""one-off (smoke test of the D32 merge): the dual t10 config with the t11 hurdle + accumulation 2 keys, cut to 20 steps,
a validation pass and a checkpoint every 10 steps; written next to this script as smoke_d32.json"""
import json
import os

here = os.path.dirname(os.path.abspath(__file__))
cfg_dir = os.path.join(here, "..", "..", "training", "configs")
c = json.load(open(os.path.join(cfg_dir, "t10_dual_s0.json")))
t11 = json.load(open(os.path.join(cfg_dir, "t11_vel_hurdle_acc2.json")))
c["model"]["object_moving"] = t11["model"]["object_moving"]
c["loss"].update({k: t11["loss"][k] for k in ("warmup_hold", "warmup_ramp")})
c.update(name="smoke_d32", steps=20, eval_every=10, save_every=10, keep=1, warmup=5, accum=t11["accum"],
         _about="smoke test of the D32 merge (2026-11-01): t10_dual_s0 + the t11 hurdle and accumulation 2, 20 steps; scratch, not a result")
json.dump(c, open(os.path.join(here, "smoke_d32.json"), "w"), indent=2)  # file name = run dir name of run_ar.sh = c["name"]
print({k: c[k] for k in ("name", "steps", "eval_every", "save_every", "keep", "accum", "eval_batches", "feature_cache", "feature_cache2", "gt_cache", "out_root")})
