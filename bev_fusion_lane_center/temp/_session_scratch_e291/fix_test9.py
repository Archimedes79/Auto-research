p = "temp/e291/test_e291_staged.py"
s = open(p, encoding="utf-8").read()
def sub(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old[:80])
    s = s.replace(old, new)
sub('prior_mode="clean,noise,pm1,u,s", prior_noise=0.3)', 'prior_mode="clean,noise,pm1,u,s,sc", prior_noise=0.3)')
sub('names = ["clean", "noise30", "pm1", "u", "s"]', 'names = ["clean", "noise30", "pm1", "u", "s", "sc"]')
sub('d = os.path.join(root, "runs", f"p_s0_{n}")', 'd = os.path.join(root, "runs", f"p_s0_{n}_max16")                                           # a --max dump has its own folder names')
sub('''    assert (n_of("clean") >= 0).any() and (n_of("u") == -1).all()''', '''    arm_of = lambda n: np.concatenate([f["prior"]["arm"] for f in fr[n]])  # noqa: E731
    assert np.array_equal(arm_of("sc"), arm_of("clean")) and np.array_equal(arm_of("u"), arm_of("clean")), "SC and U keep the geometry: the same keypoints see the same arms"
    assert (n_of("sc") >= 0).any() and not np.array_equal(n_of("sc"), n_of("clean")) and not np.array_equal(n_of("sc"), n_of("s")), "SC: the own arms with the counts of another frame"
    assert (n_of("clean") >= 0).any() and (n_of("u") == -1).all()''')
sub('''    assert json.load(open(os.path.join(root, "runs", "p_s0_clean", "ends_head.json")))["prior"]["term_over_token"] > 0
    try:
        JV.dump(argparse.Namespace(run="p_s0", thr=0.1, max=16, prior_mode="clean", prior_noise=0.0))
        raise RuntimeError("an existing read-out folder must be refused")
    except FileExistsError:
        pass''', '''    heads = {n: json.load(open(os.path.join(root, "runs", f"p_s0_{n}_max16", "ends_head.json")))["prior"] for n in names}
    assert heads["clean"]["term_over_token"] > 0 and all(0.0 <= h["arm_share"] <= 1.0 for h in heads.values()) and abs(heads["sc"]["arm_share"] - heads["clean"]["arm_share"]) < 1e-9 and heads["s"]["arm_share"] <= heads["clean"]["arm_share"]
    print("  share of the valid keypoints with an arm in reach:", {n: round(h["arm_share"], 3) for n, h in heads.items()})
    runs = lambda: sorted(os.listdir(os.path.join(root, "runs")))  # noqa: E731
    before = runs()
    try:
        JV.dump(argparse.Namespace(run="p_s0", thr=0.1, max=16, prior_mode="noise10,sc", prior_noise=0.0))                        # sc exists, noise10 does not: all names are checked first, nothing is made
        raise RuntimeError("an existing read-out folder must be refused")
    except AssertionError as e:
        assert "never overwritten" in str(e) and runs() == before, (str(e), runs())
    import lane_gt_data as LD
    real_loader = LD.loader

    def boom(*args, **kw):
        raise ValueError("a crash in the pass")
    LD.loader = boom
    try:
        JV.dump(argparse.Namespace(run="p_s0", thr=0.1, max=16, prior_mode="noise10,pm1", prior_noise=0.0))
        raise RuntimeError("the crash must propagate")
    except ValueError:
        assert runs() == before, "a crash leaves no folder (empty or not)"
    finally:
        LD.loader = real_loader''')
open(p, "w", encoding="utf-8", newline="\n").write(s)
print("ok")
