import json
for f in ("runs/e365_s1_s0/token_timing_20261117T034121Z.json", "runs/e365_s1_s0/token_timing_20261117T034328Z.json"):
    d = json.load(open(f))
    print(f.split("/")[-1], {k: d.get(k) for k in ("K", "frames", "repeats", "bf16", "quiet", "usable_for_gcost", "params", "params_bprime", "params_whole_path", "gmacs_head")})
    print("   median_ms", d.get("median_ms"), "p90_ms", d.get("p90_ms"))
    print("   host", d.get("host"))
