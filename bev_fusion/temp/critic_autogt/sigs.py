import hashlib, subprocess, csv
def show(rev, path):
    return subprocess.run(["git", "show", f"{rev}:{path}"], capture_output=True, check=True).stdout
FILES = ("grid.py", "layers.py", "flow.py", "nuscenes_adapter.py", "nuscenes_gt.py")
rows = {r["version"]: r for r in csv.DictReader(open("autogt/versions.csv", encoding="utf-8"))}
def sig_pkg(rev, base, tail):
    h = hashlib.sha1()
    for n in FILES: h.update(show(rev, f"{base}/{n}"))
    h.update(tail.encode()); return h.hexdigest()[:10]
t3 = repr((True, [], True, True, None))
t45 = repr((True, [], True, True, None, "camera"))
res = {}
for v, rev in (("v1", "7f5cb10"), ("v2", "7c8f12e")):
    h = hashlib.sha1(show(rev, "bev_fusion/training/gt_bev.py")); h.update(repr((True, [])).encode()); res[v] = (h.hexdigest()[:10], rev)
res["v3"] = (sig_pkg("autogt/v3", "bev_fusion/autogt", t3), "autogt/v3")
res["v4"] = (sig_pkg("autogt/v4", "bev_fusion/autogt", t45), "autogt/v4")
res["v5"] = (sig_pkg("autogt/v5", "bev_fusion/autogt", t45), "autogt/v5")
res["HEAD"] = (sig_pkg("HEAD", "autogt", t45), "HEAD")
for v, (s, rev) in res.items():
    print(v, rev, s, "row:", rows.get(v, {}).get("signature"), "OK" if rows.get(v, {}).get("signature") == s else "--")
# tags vs row commit
for v, r in rows.items():
    print(v, r["tag"], subprocess.run(["git", "rev-parse", "--short", r["tag"] + "^{commit}"], capture_output=True, text=True).stdout.strip(), "|", r["generator"][:80])
