import json, time, statistics
# mean GPU sm utilisation (nvidia-smi dmon, every 2 s) inside the training window of each run: from its step-15 row to its last row
samples = []
for l in open("/mnt/c/Auto-research/bev_fusion/temp/speed_dmon.txt"):
    if l.startswith("#") or not l.strip():
        continue
    p = l.split()
    try:
        h, m, s = map(int, p[0].split(":"))
        samples.append((h * 3600 + m * 60 + s, int(p[2]), int(p[3])))
    except ValueError:
        pass
print("samples", len(samples))
for name in ("speed_a", "speed_b", "speed_c2", "speed_a3"):
    rows = [json.loads(x) for x in open(f"/mnt/c/Auto-research/bev_fusion/runs/{name}/log.jsonl") if x.strip()]
    rows = [r for r in rows if "t" in r and "val" not in r]
    def sod(t):
        lt = time.localtime(t)
        return lt.tm_hour * 3600 + lt.tm_min * 60 + lt.tm_sec
    for lo_step, label in ((15, "from step 15"), (45, "from step 45")):
        rr = [r for r in rows if r["step"] >= lo_step]
        if len(rr) < 2:
            continue
        a, b = sod(rr[0]["t"]), sod(rr[-1]["t"])
        sm = [s for t, s, _ in samples if a <= t <= b]
        mem = [m for t, _, m in samples if a <= t <= b]
        if sm:
            print(f"{name:9} {label:13} {b - a:4d} s  sm util mean {statistics.mean(sm):5.1f} %  median {statistics.median(sm):5.1f} %  (mem-copy util mean {statistics.mean(mem):4.1f} %, {len(sm)} samples)")
