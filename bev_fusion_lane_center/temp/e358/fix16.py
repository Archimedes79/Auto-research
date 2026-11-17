import sys
sys.path.insert(0, "/mnt/c/Auto-research/bev_fusion_lane_center/temp/e358")
from edit_helper import apply
apply("/mnt/c/Auto-research/bev_fusion_lane_center/evaluation/jwire_check.py", [
("""            pput(card(**prim), ptwin)
            for argv in (["--read-out", "primary"], ["--ends", "att"], ["--read-out", "primary", "--ends", "att"]):
                assert main(["twin", "--run", "rx", "--seed", "0", "--tag", "twinatt"] + argv) == 0, argv
            assert main(["twin", "--run", "rx", "--seed", "0", "--tag", "twin"]) == 1, "the snapped read-out of cards that are primary ones\"""", """            pput(card(**prim), ptwin)
            with contextlib.redirect_stdout(io.StringIO()) as out:
                for argv in (["--read-out", "primary"], ["--ends", "att"], ["--read-out", "primary", "--ends", "att"]):
                    assert main(["twin", "--run", "rx", "--seed", "0", "--tag", "twinatt"] + argv) == 0, argv
                assert main(["twin", "--run", "rx", "--seed", "0", "--tag", "twin"]) == 1, "the snapped read-out of cards that are primary ones"
            assert out.getvalue().count("against the stored primary card: IDENTICAL") == 3 and "against the stored snap card: DIFFERENT" in out.getvalue(), out.getvalue()"""),
("""            with mock.patch.object(sys.modules[__name__], "live_check", lambda *x: calls.append(x) or (True, 24, {})):
                assert main(""", """            with mock.patch.object(sys.modules[__name__], "live_check", lambda *x: calls.append(x) or (True, 24, {})), contextlib.redirect_stdout(io.StringIO()):
                assert main("""),
("""def _selftest():
    import tempfile""", """def _selftest():
    import contextlib
    import io
    import tempfile"""),
])
print("ok")
