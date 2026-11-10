import glob, json, os
for f in sorted(glob.glob("runs/*/eval/door.json")):
    d = json.load(open(f))
    for c in d["calls"]:
        o = c["options"]
        rows = [(s["name"], s["seconds"]) for s in c["steps"] if s["exit"] == 0 and s["seconds"]]
        if any(n == "numbers" for n, _ in rows):
            print(f.split("/")[1], "device", o.get("device"), "n", o.get("n"), "workers", o.get("workers"), "|", ", ".join(f"{n} {s:.0f}s" for n, s in rows))
for f in sorted(glob.glob("runs/*/eval/latency.json"))[:6]:
    d = json.load(open(f))
    lat = d.get("latency", {})
    print(f.split("/")[1], {k: lat[k] for k in lat if "ms" in k or k in ("gpu",)} if isinstance(lat, dict) else lat)
