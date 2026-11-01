import sys
p = sys.argv[1]
s = open(p, encoding="utf-8").read()
def rep(old, new, count=1):
    global s
    assert s.count(old) == count, (old[:70], s.count(old))
    s = s.replace(old, new)
rep('''    assert DC.decide([0.80, 0.69], [0.70, 0.70])[0] == "INCONCLUSIVE", "the typed interface agrees"''',
'''    assert DC.decide([0.80, 0.69], [0.70, 0.70])[0] == "INCONCLUSIVE", "the typed interface agrees"
    # ... but a guard-only headline on subset-only cards is judged by the guard's subsets, not flagged as disagreeing seeds (reviewer finding)
    json.dump(dict(n=dict(a=dict(Q=0.80), c=dict(Q=0.70)), b=dict(a=dict(Q=0.80), c=dict(Q=0.70)), x=dict(a=dict(Q=0.69), c=dict(Q=0.70))), open(f"{T}/subq.json", "w"))
    subq = dict(a=dict(cards={k: f"subq.json#{k}/a" for k in "nbx"}), c=dict(cards={k: f"subq.json#{k}/c" for k in "nbx"}))
    assert verdict(arms=subq, rule=[dict(metric="Q", max_loss=0.05, subsets=["n", "b", "x"], at_least=2)]) == "KEEP", "subset-only guard: no seed flags"
    # lower-is-better names: project.json "lower_is_better" replaces the legacy lane list, which only a problem with tag prefix lcr gets
    assert "vert100" in DC.lower_names(T)
    pj = json.load(open(f"{T}/project.json"))
    json.dump(dict(pj, lower_is_better=[]), open(f"{T}/project.json", "w"))
    assert "vert100" not in DC.lower_names(T) and "ms" in DC.lower_names(T), "an empty list: vert100 reads higher-is-better"
    json.dump({k: v for k, v in pj.items() if k != "lower_is_better"}, open(f"{T}/project.json", "w"))
    assert "vert100" not in DC.lower_names(T), "no key, tag prefix t: no lane names"
    json.dump(dict({k: v for k, v in pj.items() if k != "lower_is_better"}, tag_prefix="lcr"), open(f"{T}/project.json", "w"))
    assert "vert100" in DC.lower_names(T), "no key, the lane problem: the legacy list until its key is written"
    json.dump(pj, open(f"{T}/project.json", "w"))''')
rep('''    own = go("S.e124", ["bash", "-c", "exit 124"], "--no-git")''',
'''    envrun = go("S.env", ["bash", "-c", 'echo "id=$AR_RUN_ID commit=$AR_COMMIT tag=$AR_TAG"'])
    rec = json.load(open(f"{T}/runs/S.env/run.json"))
    assert envrun.returncode == 0 and f"id=S.env commit={rec['commit']} tag=t/S.env" in open(f"{T}/runs/S.env/run.log").read(), open(f"{T}/runs/S.env/run.log").read()
    own = go("S.e124", ["bash", "-c", "exit 124"], "--no-git")''')
open(p, "w", encoding="utf-8", newline="").write(s)
print("ok")
