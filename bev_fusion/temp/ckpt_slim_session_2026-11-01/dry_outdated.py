"""read-only: which flow caches in temp/checks would scratch.cached() call outdated under the folder rules written into the three tests"""
import glob, os, sys, time
sys.path.insert(0, "training")
import scratch

RULES = {"flow_timed_nusc": ["data/cache/gt"], "flow_fused_nusc": ["data/cache/gt", "data/cache/radar"],
         "flow_explains_nusc": ["data/cache/gt"], "flow_explains_nusc_v1": ["data/cache/gt"]}
stamp = lambda t: time.strftime("%m-%d %H:%M", time.localtime(t))
t0 = time.time()
for p in sorted(glob.glob("temp/checks/flow_*_nusc*_*.pt")):
    name = os.path.basename(p)
    kind = next((k for k in sorted(RULES, key=len, reverse=True) if name.startswith(k + "_")), None)
    if kind is None:
        print(f"{name:55s} no rule"); continue
    newest = scratch.newest_mtime(RULES[kind])
    verdict = "OUTDATED (a dep folder is newer)" if os.path.getmtime(p) < newest else "valid"
    print(f"{name:55s} {os.path.getsize(p)/1e6:7.0f} MB  cache {stamp(os.path.getmtime(p))}  newest dep {stamp(newest)}  {verdict}")
print(f"{time.time()-t0:.1f} s")
