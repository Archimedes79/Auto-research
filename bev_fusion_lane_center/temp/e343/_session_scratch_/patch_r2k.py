import os
R = os.environ["BFLC_ROOT"]
p = os.path.join(R, "evaluation", "tests", "test_jtest_eval.py")
s = open(p, encoding="utf-8").read()
pairs = [
    ('''def run_toy_seed(d, xs, card_base=0.5, ctl_head=0.6, stack=True, split="val", ds=None, card_curves=7, card_wired=5):''',
     '''def run_toy_seed(d, xs, card_base=0.5, ctl_head=0.6, stack=True, split="val", ds=None, card_curves=7, card_wired=5, b_device="cpu", ctl_prov=None):'''),
    ('''    for name, run in (("predfull_e273_p_s0", "e273_p_s0"), ("ends_e324_s_s0", "e324_s_s0"), ("predfull_e292_c2_s0", "e292_c2_s0")):
        pickle.dump(full, open(os.path.join(d, name + ".pkl"), "wb"))
        json.dump(dict(provenance=PROV(run, "cpu")), open(os.path.join(d, name + "_head.json"), "w"))
    pickle.dump(full, open(os.path.join(d, "ctl_ends.pkl"), "wb"))''',
     '''    for name, run in (("predfull_e273_p_s0", "e273_p_s0"), ("ends_e324_s_s0", "e324_s_s0"), ("predfull_e292_c2_s0", "e292_c2_s0")):
        pickle.dump(full, open(os.path.join(d, name + ".pkl"), "wb"))
        json.dump(dict(provenance=PROV(run, b_device if run == "e273_p_s0" else "cpu")), open(os.path.join(d, name + "_head.json"), "w"))
    pickle.dump(full, open(os.path.join(d, "ctl_ends.pkl"), "wb"))
    if os.path.exists(os.path.join(d, "ctl_ends_head.json")):
        os.remove(os.path.join(d, "ctl_ends_head.json"))                                              # a sidecar of an earlier call of this helper in the same folder (scratch of this test only)
    if ctl_prov:
        json.dump(dict(provenance=ctl_prov), open(os.path.join(d, "ctl_ends_head.json"), "w"))'''),
    ('''        r, _ = run_toy_seed(d, XS, split="test", card_base=0.5 + 1e-10)
        assert r["control"]["plain_check"]["ok"] and r["control"]["repro"]["ok"]
''', '''        r, _ = run_toy_seed(d, XS, split="test", card_base=0.5 + 1e-10)
        assert r["control"]["plain_check"]["ok"] and r["control"]["repro"]["ok"]
        # provenance of dumps named by a flag or made earlier: B's dump on TEST is not made, a sidecar that says another device refuses it; the control's pair dump likewise; the default dumps need theirs
        assert raises(SystemExit, run_toy_seed, d, XS, split="test", b_device="cuda"), "B's dump (flagged / not made) was made on another device: refused"
        assert raises(SystemExit, run_toy_seed, d, XS, ctl_prov=PROV("e312_p_s0", "cuda")) and raises(SystemExit, run_toy_seed, d, XS, ctl_prov=PROV("e312_p_s1", "cpu")), "the control's pair dump of another device / run: refused"
        r, _ = run_toy_seed(d, XS, ctl_prov=PROV("e312_p_s0", "cpu"))
        assert r["control"]["repro"]["ok"], "the control's pair dump with its own provenance passes"
'''),
]
for old, new in pairs:
    assert s.count(old) == 1, old[:70]
    s = s.replace(old, new)
open(p, "w", encoding="utf-8").write(s)
