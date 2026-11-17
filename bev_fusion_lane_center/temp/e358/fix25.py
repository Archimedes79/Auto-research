import sys
sys.path.insert(0, "/mnt/c/Auto-research/bev_fusion_lane_center/temp/e358")
from edit_helper import apply
apply("/mnt/c/Auto-research/bev_fusion_lane_center/evaluation/jwire_check.py", [
("""            assert main(["twin", "--run", "rx", "--seed", "0", "--tag", "twin"]) == 1, "the snapped read-out of cards that are primary ones"
""", """            assert main(["twin", "--run", "rx", "--seed", "0", "--tag", "twin"]) == 1, "the snapped read-out of cards that are primary ones"
            for argv in (["twin", "--run", "rx", "--seed", "0", "--tag", "t", "--ends", "bogus"], ["live", "--seed", "0", "--pair-run", "rx", "--ends", "bogus"], ["live", "--seed", "0", "--pair-run", "rx", "--read-out", "bogus"]):
                with contextlib.redirect_stderr(io.StringIO()):
                    try:
                        main(argv)
                        raise AssertionError(f"{argv} must be refused by the command line")
                    except SystemExit as ex:
                        assert ex.code == 2, argv
"""),
])
apply("/mnt/c/Auto-research/bev_fusion_lane_center/evaluation/jwire_verdict.py", [
("""    print("jwire_verdict selftest ok")""", """    # the command line: the default read-out is the snapped one (E325 / E350 unchanged), --read-out primary reads the primary cards
    global ROOT
    old_root, io_out = ROOT, __import__("io")
    try:
        for ro, args, want in (("snap", [], "KEEP+"), ("primary", ["--read-out", "primary"], "KEEP+"), ("snap", ["--read-out", "primary"], "INCOMPLETE"), ("primary", [], "INCOMPLETE")):
            with tempfile.TemporaryDirectory() as t:
                world(t, {0: gp if ro == "primary" else good, 1: gp if ro == "primary" else good}, "cs", ro)
                ROOT = t
                with __import__("contextlib").redirect_stdout(io_out.StringIO()) as buf:
                    main(["--card", "E000", "--control-prefix", "cs", "--head-prefix", "hs", "--tag", "tw", "--boot", "temp/e000/boot_w_vs_c_s{seed}.log"] + args)
                assert buf.getvalue().splitlines()[-1].startswith(f"TOKEN (E000): {want}"), (ro, args, buf.getvalue().splitlines()[-1])
    finally:
        ROOT = old_root
    print("jwire_verdict selftest ok")"""),
])
apply("/mnt/c/Auto-research/bev_fusion_lane_center/temp/_session_scratch_e358/mutate.py", [
("""        ("run_pipeline: wn judged on the jittered ends", "at = agree(predict(w, nf, nm), nl, no, nn)", "at = agree(predict(w, feat, mask), label, orc, npl)"),""", """        ("run_pipeline: wn judged on the jittered ends", "agree(predict(w, nf, nm), nl, no, nn)", "agree(predict(w, feat, mask), label, orc, npl)"),"""),
])
print("ok")
