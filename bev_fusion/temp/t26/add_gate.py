gate = open("/mnt/c/Auto-research/bev_fusion/temp/t26/ram_gate_lines.txt").read()
for f in ("job_base_eval43.sh", "job_ref_maps_base.sh"):
    p = "/mnt/c/Auto-research/bev_fusion/temp/t26/" + f
    s = open(p).read()
    if "MemAvailable" in s:
        print(f, "already gated"); continue
    marker = "export PYTHONPATH"
    i = s.index(marker); j = s.index("\n", i) + 1
    open(p, "w").write(s[:j] + gate + s[j:]); print(f, "gated")
