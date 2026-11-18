import sys
sys.path.insert(0, "/mnt/c/Auto-research/bev_fusion_lane_center/temp/e358")
from edit_helper import apply
apply("/mnt/c/Auto-research/bev_fusion_lane_center/temp/_session_scratch_e358/test_staged.py", [
("""    JW._selftest() if hasattr(JW, "_selftest") else JW.selftest()""", """    JW._selftest()"""),
("""import hashlib
import os""", """import contextlib
import hashlib
import os"""),
("""def sha(path):""", """@contextlib.contextmanager
def project_root(d):
    \"\"\"a temp project root for the code that reads `ROOT` at call time: `QE.ROOT` and the environment variable `jwire_head.load_wire` reads\"\"\"
    old_root, old_env = QE.ROOT, os.environ.get("BFLC_ROOT")
    QE.ROOT = os.environ["BFLC_ROOT"] = d
    try:
        yield
    finally:
        QE.ROOT = old_root
        if old_env is None:
            os.environ.pop("BFLC_ROOT", None)
        else:
            os.environ["BFLC_ROOT"] = old_env


def sha(path):"""),
("""        old = os.environ.get("BFLC_ROOT")
        os.environ["BFLC_ROOT"] = d
        try:
            h1, h0 = JH.load_wire("hd", "model_h1.pt"), JH.load_wire("hd")
        finally:
            os.environ["BFLC_ROOT"] = old
""", """        with project_root(d):
            h1, h0 = JH.load_wire("hd", "model_h1.pt"), JH.load_wire("hd")
"""),
("""    old_root, old_env = QE.ROOT, os.environ.get("BFLC_ROOT")
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "runs", "e324_s_s0"))""", """    with tempfile.TemporaryDirectory() as d, project_root(d):
        os.makedirs(os.path.join(d, "runs", "e324_s_s0"))"""),
("""        QE.ROOT = d
        os.environ["BFLC_ROOT"] = d
        try:
            def score(head, attach, wire="head", seed=0, run="e324_s_s0", tag="t"):
                try:
                    QE.cmd_score(NS(**dict(opts, seed=seed, run=run, tag=tag, wire=wire, wire_run=head, attach=attach)))
                except AssertionError as e:
                    return "refused: " + str(e)
                except FileNotFoundError as e:
                    return "passed: " + os.path.basename(str(e.filename))
                return "ran"
            assert score("h_att", True) == "passed: model.pt" and score("h_att:model_wn.pt", True) == "passed: model.pt", "an att head on its own pair run and seed"
            assert score("h_kp350", False) == "passed: model.pt" and score("h_legacy", False, run="e312_p_s0") == "passed: model.pt", "kp heads on the keypoint read-out (E350: its pair run; E325: E312's)"
            for head, attach, why in (("h_kp350", True, "trained on 'kp' ends"), ("h_legacy", True, "trained on 'kp' ends"), ("h_att", False, "trained on 'att' ends"), ("h_att_other", True, "pair run"), ("h_att_s1", True, "seed 1"), ("h_legacy", False, "pair run")):
                r = score(head, attach)
                assert r.startswith("refused") and why in r, (head, attach, r)
            assert "seed" in score("h_att", True, seed=1) and score("h_att", True, wire="twin", head=None) == "passed: model.pt", "the twin has no head to check"
        finally:
            QE.ROOT = old_root
            os.environ["BFLC_ROOT"] = old_env
""", """
        def score(head, attach, wire="head", seed=0, run="e324_s_s0", tag="t"):
            try:
                QE.cmd_score(NS(**dict(opts, seed=seed, run=run, tag=tag, wire=wire, wire_run=head, attach=attach)))
            except AssertionError as e:
                return "refused: " + str(e)
            except FileNotFoundError as e:
                return "passed: " + os.path.basename(str(e.filename))
            return "ran"
        assert score("h_att", True) == "passed: model.pt" and score("h_att:model_wn.pt", True) == "passed: model.pt", "an att head on its own pair run and seed"
        assert score("h_kp350", False) == "passed: model.pt" and score("h_legacy", False, run="e312_p_s0") == "passed: model.pt", "kp heads on the keypoint read-out (E350: its pair run; E325: E312's)"
        for head, attach, why in (("h_kp350", True, "trained on 'kp' ends"), ("h_legacy", True, "trained on 'kp' ends"), ("h_att", False, "trained on 'att' ends"), ("h_att_other", True, "pair run"), ("h_att_s1", True, "seed 1"), ("h_legacy", False, "pair run")):
            r = score(head, attach)
            assert r.startswith("refused") and why in r, (head, attach, r)
        assert "seed" in score("h_att", True, seed=1) and score(None, True, wire="twin") == "passed: model.pt", "the twin has no head to check"
"""),
])
print("ok")
