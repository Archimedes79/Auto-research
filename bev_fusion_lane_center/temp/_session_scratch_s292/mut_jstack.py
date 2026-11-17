import io
import subprocess

src = io.open("evaluation/jstack.py", encoding="utf-8").read()
M = [
    ("swap: no key-set assertion", 'assert src and set(src) == set(mine), f"node-head keys differ: {sorted(set(src) ^ set(mine))[:6]}"', "assert src"),
    ("swap: no shape assertion", 'assert all(src[k].shape == mine[k].shape for k in src), "node-head shapes differ"', "pass"),
    ("swap: changed counted wrong", "sum(not torch.equal(src[k], mine[k]) for k in src)", "sum(torch.equal(src[k], mine[k]) for k in src)"),
    ("swap: the pair state is modified in place", "    new = dict(state)\n    new.update(src)\n    return new, len(src)", "    new = state\n    new.update(src)\n    return new, len(src)"),
    ("pyramid: values not compared", "all(k2 in state and torch.equal(node_state[k], state[k2]) for k, k2 in zip(pyr, other))", "all(k2 in state for k, k2 in zip(pyr, other))"),
    ("pyramid: empty accepted", "return bool(pyr) and all(", "return all("),
    ("same: None equals anything", "return (u is None) == (v is None)", "return True"),
    ("same: NaN unequal", "equal_nan=True)", "equal_nan=False)"),
    ("compare: row not compared", 'if x["row"] != y["row"] or not same(x.get(k), y.get(k))', 'if not same(x.get(k), y.get(k))'),
    ("layer: e not damped", 'no["e"] = QMN.matrix_nms_logit(no["e"], no["pos"], self.radius, self.sigma)', 'pass'),
    ("layer: e_raw not kept", 'no["e_raw"] = no["e"]\n', "\n"),
    ("layer: keys dropped", "no = dict(self.nodes(*a))", "no = dict(e=self.nodes(*a)['e'], pos=self.nodes(*a)['pos'])"),
    ("layer params: no c2 assertion", 'assert not sn, f"runs/e292_{g}_s{seed} has a layer {sn}"', "pass"),
    ("layer params: no mnms assertion", 'assert sn and sn[0] == "mnms" and len(sn) == 3, f"runs/e292_a_s{seed}: soft_nms {sn}"', "pass"),
    ("layer params: radius / sigma swapped", "return float(sn[1]), float(sn[2])", "return float(sn[2]), float(sn[1])"),
    ("model_for: wraps always", "    if lp is not None:\n        m.nodes = LayerNodes", "    if True:\n        m.nodes = LayerNodes"),
    ("model_for: unknown graph accepted", "    assert g in GRAPHS, run\n", ""),
]
killed = 0
for name, a, b in M:
    if src.count(a) != 1:
        print(f"NOT APPLIED  {name}: {src.count(a)} matches")
        continue
    io.open("temp/_session_scratch_e358/mut_bs/jstack.py", "w", encoding="utf-8").write(src.replace(a, b))
    r = subprocess.run(["bash", "tools/env/_py14.sh", "temp/_session_scratch_e358/mut_bs/jstack.py", "selftest"], capture_output=True, text=True)
    if r.returncode != 0:
        killed += 1
        print(f"killed       {name}")
    else:
        print(f"SURVIVED     {name}")
print(f"{killed} of {len(M)} mutants killed")
