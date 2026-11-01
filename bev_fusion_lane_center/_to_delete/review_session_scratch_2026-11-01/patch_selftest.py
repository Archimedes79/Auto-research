import sys
p = sys.argv[1]
s = open(p, encoding="utf-8").read()
def rep(old, new, count=1):
    global s
    assert s.count(old) == count, (old[:70], s.count(old))
    s = s.replace(old, new)
rep('''               code_dirs=["code"], scratch="temp", live_code=["code"], cpu_slots=5), open(f"{T}/project.json", "w"))''',
    '''               code_dirs=["code"], scratch="temp", live_code=["code"], cpu_slots=5, lower_is_better=["vert100"]), open(f"{T}/project.json", "w"))''')
rep('''    shutil.rmtree(f"{T}/runs/.gpu_slots")
    # a slot directory without an owner file''', '''    shutil.rmtree(f"{T}/runs/.gpu_slots")
    # project.json gpu_slots_dir: one slot folder shared by the problems of a machine (D23); a live slot there blocks this problem too
    shared = tempfile.mkdtemp(prefix="ar_slots_")
    assert RUN.slot_dir(T, dict(gpu_slots_dir=shared), f"{T}/runs") == os.path.normpath(shared) and RUN.slot_dir(T, {}, f"{T}/runs") == f"{T}/runs/.gpu_slots"
    json.dump(dict(json.load(open(f"{T}/project.json")), gpu_slots_dir=shared), open(f"{T}/project.json", "w"))
    os.makedirs(f"{shared}/0")
    json.dump(dict(id="other", pid=os.getpid(), run_dir=f"{T}/runs/S.live"), open(f"{shared}/0/run.json", "w"))
    busy2 = go("S.busy2", ["bash", "-c", "echo gpu"], "--gpu", "--gpu-slots", "1", "--wait-gpu", "0")
    assert busy2.returncode == 1 and rows()["S.busy2"]["status"] == "gpu_busy", busy2.stdout
    import tick as TK
    assert TK.free_gpu_slots(f"{T}/runs", 2, shared) == 1 and TK.free_gpu_slots(f"{T}/runs", 2) == 2
    shutil.rmtree(shared)
    json.dump({k: v for k, v in json.load(open(f"{T}/project.json")).items() if k != "gpu_slots_dir"}, open(f"{T}/project.json", "w"))
    # a command's own exit 124 is a failure, not the launcher's timeout (review 2026-10-31)
    own = go("S.e124", ["bash", "-c", "exit 124"], "--no-git")
    assert own.returncode == 124 and rows()["S.e124"]["status"] == "failed", (own.stdout, rows()["S.e124"]["status"])
    # a slot directory without an owner file''')
rep('''    assert verdict(arms=lower, metric="err", direction="higher", rule=[dict(metric="err", min_effect=0.01)]) == "CUT", "the same numbers read as higher-is-better"''',
'''    assert verdict(arms=lower, metric="err", direction="higher", rule=[dict(metric="err", min_effect=0.01)]) == "CUT", "the same numbers read as higher-is-better"
    # a headline judged only by a guard (no min_effect clause): its evidence still counts (review 2026-10-31, T1)
    guard = [dict(metric="Q", max_loss=0.005)]
    json.dump(dict(a=dict(Q=0.76, F=0.9), b=dict(Q=0.74, F=0.9), ca=dict(Q=0.73, F=0.9), cb=dict(Q=0.72, F=0.9)), open(f"{T}/cards.json", "w"))
    assert verdict(arms=dict(a=dict(cards=["cards.json#a"]), c=dict(cards=["cards.json#ca"])), rule=guard) == "INCONCLUSIVE", "guard-only headline, one seed"
    assert verdict(arms=two, rule=guard) == "KEEP", "guard-only headline, two agreeing seeds"
    json.dump(dict(a=dict(Q=0.80, F=0.9), b=dict(Q=0.69, F=0.9), ca=dict(Q=0.70, F=0.9), cb=dict(Q=0.70, F=0.9)), open(f"{T}/cards.json", "w"))
    assert verdict(arms=two, rule=guard) == "INCONCLUSIVE", "guard-only headline, seeds disagree"
    assert DC.decide([0.80, 0.69], [0.70, 0.70])[0] == "INCONCLUSIVE", "the typed interface agrees"
    json.dump(dict(a=dict(Q=0.76, F=0.9), b=dict(Q=0.74, F=0.9), ca=dict(Q=0.73, F=0.9), cb=dict(Q=0.72, F=0.9)), open(f"{T}/cards.json", "w"))''')
open(p, "w", encoding="utf-8", newline="").write(s)
print("ok")
