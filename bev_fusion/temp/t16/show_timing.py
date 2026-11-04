"""scratch: print the stored timing numbers of the t16 small runs (inference_ms total, params, train micro-batch)"""
import json

for r in ("t16_small_s0", "t16_small_ctl_s0", "t16_small_bg1_s0", "t16_small_clip_s0", "t16_small_final_s0"):
    p = f"bev_fusion/runs/{r}/eval/timing_modules.json"
    try:
        d = json.load(open(p))
    except OSError:
        print(r, "no timing file")
        continue
    inf = d["inference_ms"]
    print(r, "step", d["step"], "load", d.get("load"))
    print("   inference_ms:", json.dumps(inf)[:400])
    print("   params:", json.dumps(d["params"])[:200], "| train_microbatch_ms:", json.dumps(d["train_microbatch_ms"])[:200])
