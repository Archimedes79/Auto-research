p = "temp/e291/test_e291_staged.py"
s = open(p, encoding="utf-8").read()
def sub(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old[:80])
    s = s.replace(old, new)
# SCR: made when absent (the installed test has no stage folder)
sub('SCR = os.path.join(ROOT, "temp", "_session_scratch_e291")\n', 'SCR = os.path.join(ROOT, "temp", "_session_scratch_e291")\nos.makedirs(SCR, exist_ok=True)\n')
# shuffle: SC
sub('''    ds, items = _batch_items((JP.Spec("clean"), JP.Spec.readout("S")), n=12)
    ds2, items2 = _batch_items((JP.Spec.readout("S"),), n=12)''', '''    ds, items = _batch_items((JP.Spec("clean"), JP.Spec.readout("S"), JP.Spec.readout("SC")), n=12)
    ds2, items2 = _batch_items((JP.Spec.readout("S"), JP.Spec.readout("SC")), n=12)''')
sub('''        c, s = it["arms"]
        assert all(np.array_equal(s[k], it2["arms"][0][k]) for k in JP.KEYS), "S is deterministic"''', '''        c, s, sc = it["arms"]
        assert all(np.array_equal(s[k], it2["arms"][0][k]) and np.array_equal(sc[k], it2["arms"][1][k]) for k in JP.KEYS), "S and SC are deterministic"''')
sub('''        assert all(np.array_equal(s[k], want[k]) for k in JP.KEYS), "S is the clean table of the partner"
''', '''        assert all(np.array_equal(s[k], want[k]) for k in JP.KEYS), "S is the clean table of the partner"
        assert np.array_equal(sc["c"], c["c"]) and np.array_equal(sc["t"], c["t"]), "SC keeps the frame's own arm centres and axes"
        k_ = len(want["n_in"])
        assert all(np.array_equal(sc[k], want[k][np.arange(len(c[k])) % k_]) for k in ("n_in", "n_out")), "SC: arm i gets the count of arm i modulo the number of arms of the partner"
        sc_diff += len(c["n_in"]) > 0 and not (np.array_equal(sc["n_in"], c["n_in"]) and np.array_equal(sc["n_out"], c["n_out"]))
''')
sub("    diff = n_arm = 0\n", "    diff = n_arm = sc_diff = 0\n")
sub('''    print("shuffle ok:", diff, "of 12 differ")''', '''    assert n_arm >= 6 and sc_diff >= 0.6 * n_arm, f"the SC counts must differ from the clean ones on most frames with arms: {sc_diff} of {n_arm}"
    print("shuffle ok:", diff, "of 12 differ; SC counts differ in", sc_diff, "of", n_arm, "frames with arms")''')
# identity_real: skip without the reference
sub('''    ref = torch.load(os.path.join(SCR, "ref_e312.pt"))
    m = _e312_model(True)''', '''    if not os.path.exists(os.path.join(SCR, "ref_e312.pt")):
        print("identity_real SKIPPED: no ref_e312.pt (it is made by `--only ref` with the code BEFORE the install; after the install the pinned sums of jends.py's selftest take its place)")
        return
    ref = torch.load(os.path.join(SCR, "ref_e312.pt"))
    m = _e312_model(True)''')
open(p, "w", encoding="utf-8", newline="\n").write(s)
print("ok")
