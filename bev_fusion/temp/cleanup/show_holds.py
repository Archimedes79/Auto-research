"""Scratch: for each held run print ledger row fields (cut), run.json command / config, STOPPED file. Read-only."""
import csv, json, os, sys

ROOT = "/mnt/c/Auto-research/bev_fusion/records"
IDS = ["profile.t10", "r2.raster64_s0", "smoke.t22_B_s0", "smoke.t22_B_s0b", "t10.dual_long_s0", "t20.resume_s0", "t21.rec_s0"]
rows = {r["id"]: r for r in csv.DictReader(open(f"{ROOT}/LEDGER.csv", encoding="utf-8"))}
cut = lambda s, n=450: (s or "")[:n].replace("\n", " ")
for i in IDS:
    print("=" * 100)
    print(i)
    r = rows.get(i)
    if r:
        for k in ("hyp", "pred", "rule", "result", "decision", "notes", "status", "commit"):
            if k in r and r[k]:
                print(f"  ledger.{k}: {cut(r[k])}")
    else:
        print("  (no ledger row)")
    d = f"{ROOT}/runs/{i}"
    for fn in ("run.json", "STOPPED"):
        p = f"{d}/{fn}"
        if os.path.isfile(p):
            t = open(p, encoding="utf-8", errors="replace").read()
            if fn == "run.json":
                try:
                    j = json.loads(t)
                    keep = {k: j[k] for k in j if k in ("cmd", "command", "config", "warm", "state", "exit", "started", "ended", "commit", "tag", "hyp")}
                    print("  run.json:", cut(json.dumps(keep, ensure_ascii=False), 700))
                except Exception as e:
                    print("  run.json unreadable", e)
            else:
                print("  STOPPED:", cut(t, 300))
    print("  files:", sorted(os.listdir(d))[:25] if os.path.isdir(d) else "(no dir)")
