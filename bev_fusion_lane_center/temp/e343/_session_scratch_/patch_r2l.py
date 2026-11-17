import os
R = os.environ["BFLC_ROOT"]
p = os.path.join(R, "evaluation", "tests", "test_jtest_eval.py")
s = open(p, encoding="utf-8").read()
pairs = [
    ('''card_curves=7, card_wired=5, b_device="cpu", ctl_prov=None):''', '''card_curves=7, card_wired=5, b_device="cpu", ctl_prov=None, ctl_b_device=None):'''),
    ('''    a = argparse.Namespace(split=split, max=0, device="cpu", run_prefix="e324_s_s", b_dump=None, control_ends_dump=os.path.join(d, "ctl_ends.pkl"), control_card=toy_card_file(d, ctl_head, card_base, card_curves, card_wired))
    with patched((J, "dump_dir", lambda split, n=0: d),''', '''    card = toy_card_file(d, ctl_head, card_base, card_curves, card_wired)
    a = argparse.Namespace(split=split, max=0, device="cpu", run_prefix="e324_s_s", b_dump=None, control_ends_dump=os.path.join(d, "ctl_ends.pkl"), control_card=card)
    defaults = []
    if ctl_b_device:                                                                                   # the control's graph is E312's own B dump (the split's defaults), not the winner's
        pickle.dump(full, open(os.path.join(d, "ctl_b.pkl"), "wb"))
        json.dump(dict(provenance=PROV("e273_p_s0", ctl_b_device)), open(os.path.join(d, "ctl_b_head.json"), "w"))
        a.control_ends_dump = a.control_card = None
        defaults = [(J, "CONTROL_DEFAULTS", {split: dict(ends=os.path.join(d, "ctl_ends.pkl"), b=os.path.join(d, "ctl_b.pkl"), card=card)})]
    with patched(*defaults, (J, "dump_dir", lambda split, n=0: d),'''),
    ('''        r, _ = run_toy_seed(d, XS, ctl_prov=PROV("e312_p_s0", "cpu"))
        assert r["control"]["repro"]["ok"], "the control's pair dump with its own provenance passes"
''', '''        r, _ = run_toy_seed(d, XS, ctl_prov=PROV("e312_p_s0", "cpu"))
        assert r["control"]["repro"]["ok"], "the control's pair dump with its own provenance passes"
        assert run_toy_seed(d, XS, ctl_b_device="cpu")[0]["control"]["repro"]["ok"] and raises(SystemExit, run_toy_seed, d, XS, ctl_b_device="cuda"), "the control's own B dump (the split's default) is checked too"
'''),
]
for old, new in pairs:
    assert s.count(old) == 1, old[:70]
    s = s.replace(old, new)
open(p, "w", encoding="utf-8").write(s)
m = os.path.join(R, "temp", "e343", "_session_scratch_", "mutate.py")
t = open(m, encoding="utf-8").read()
old = ''' ("get_dump no provenance", '    elif expect is not None:\n        check_provenance(path, expect, what)', '    elif False:\n        check_provenance(path, expect, what)'),'''
new = ''' ("get_dump no provenance", '    elif expect is not None and (make is not None or os.path.exists(head_path(path))):', '    elif False:'),'''
assert t.count(old) == 1, t.count(old)
open(m, "w", encoding="utf-8").write(t.replace(old, new))
print("ok")
