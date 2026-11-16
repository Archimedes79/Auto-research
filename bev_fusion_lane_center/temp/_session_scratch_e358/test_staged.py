"""E358 scratch: tests of the STAGED reading-side copies (`stage/qends_eval.py`, `stage/jends_eval.py`, `stage/jwire_head.py`) of E343's frozen files:

    wsl bash temp/e358/run_scratch.sh temp/_session_scratch_e358/test_staged.py        # gcar env, CPU, nice, 2 threads (the wrapper puts the stage dir first on the path)

What is pinned: the staged copies are what is imported (and the originals in the tree are byte for byte the ones whose sha256 was recorded), `check_wire` accepts `--attach` with `--snap` for twin / head and still refuses the
others, `JH.load_wire` keeps the checkpoint args, `JH.check_head` refuses a head of the wrong end kind / seed / pair run (no field = kp / E312's run), `cmd_score` refuses before anything long (through the real call), the staged scorer
takes `--attach --snap --snap-nogeom --wire head` on its command line, and the staged copies still pass their own selftests next to the original tree modules (`jwire`, `qends`, ...).
"""
import contextlib
import hashlib
import os
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS

HERE = os.path.dirname(os.path.abspath(__file__))
STAGE = os.path.join(HERE, "stage")
ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, STAGE)                                                                             # the staged copies first, then the tree
import torch  # noqa: E402
import qends_eval as QE  # noqa: E402
import jwire_head as JH  # noqa: E402
import jends_eval as JE  # noqa: E402
import qgraph_eval  # noqa: E402,F401  (`cmd_score` imports them: before a test points BFLC_ROOT at a temp folder, which their own path set-up reads)
import qgraph_readout  # noqa: E402,F401


def refused(f, *a, **k):
    try:
        f(*a, **k)
    except AssertionError as e:
        return str(e) or "refused"
    return None


@contextlib.contextmanager
def project_root(d):
    """a temp project root for the code that reads `ROOT` at call time: `QE.ROOT` and the environment variable `jwire_head.load_wire` reads"""
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


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def test_staged_is_imported_and_originals_are_untouched():
    for mod in (QE, JH, JE):
        assert os.path.dirname(os.path.abspath(mod.__file__)) == STAGE, mod.__file__
    for line in open(os.path.join(STAGE, "sha256_of_originals.txt")):
        h, rel = line.split(" *")
        rel = rel.strip()
        assert sha(os.path.join(ROOT, rel)) == h, f"{rel} of the tree changed (E343's digest)"
        assert sha(os.path.join(STAGE, os.path.basename(rel))) != h, f"{rel} has no staged difference"


def test_check_wire():
    ok = [NS(), NS(wire="twin", snap=True), NS(wire="head", snap=True, wire_run="x"), NS(wire="twin", snap=True, attach=True, snap_nogeom=True), NS(wire="head", snap=True, attach=True, snap_nogeom=True, wire_run="x:model_wn.pt")]
    assert [QE.check_wire(a) for a in ok] == [(None, None), ("twin", None), ("head", "x"), ("twin", None), ("head", "x:model_wn.pt")]
    bad = [NS(wire="twin"), NS(wire="twin", attach=True, snap_nogeom=True), NS(wire="twin", snap=True, attach=True), NS(wire="head", snap=True, attach=True, wire_run="x"), NS(wire="head", snap=True, attach=True, snap_nogeom=False, wire_run="x"), NS(wire="head", attach=True, snap_nogeom=True, wire_run="x"), NS(wire="head", snap=True, attach=True, snap_nogeom=True), NS(wire="twin", snap=True, attach=True, snap_nogeom=True, wire_run="x"), NS(wire="twin", snap=True, attach=True, snap_nogeom=True, attach_r=1.5),
           NS(wire="twin", snap=True, attach=True, snap_nogeom=True, snap_all=True), NS(wire="head", snap=True, attach=True, snap_nogeom=True, snap_guard=True, wire_run="x"), NS(wire="soft", snap=True), NS(snap=True, attach=True, snap_nogeom=True, wire_run="x")]
    for a in bad:
        assert refused(QE.check_wire, a), a


def test_load_wire_keeps_the_args():
    w = JH.Wire.rule_like()
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "runs", "hd"))
        args = dict(seed=0, pair_run="e324_s_s0", ends="att", head_seed=1)
        torch.save(dict(state=w.state_dict(), args=args), os.path.join(d, "runs", "hd", "model_h1.pt"))
        torch.save(dict(state=w.state_dict()), os.path.join(d, "runs", "hd", "model.pt"))                       # a head without args (nothing recorded)
        with project_root(d):
            h1, h0 = JH.load_wire("hd", "model_h1.pt"), JH.load_wire("hd")
        assert h1.meta == args and h0.meta == {} and not h1.training
        assert all(torch.equal(p, q) for p, q in zip(h1.state_dict().values(), w.state_dict().values()))


def test_check_head():
    head = lambda **m: NS(meta=m)  # noqa: E731
    legacy, kp350, att = head(), head(seed=0, pair_run="e324_s_s0"), head(seed=0, pair_run="e324_s_s0", ends="att")
    JH.check_head(legacy, "e312_p_s0", 0, False)                                                          # E325's head: no field = kp, E312's run
    JH.check_head(legacy, "e312_p_s1", 1, False)                                                          # ... of the seed of the read-out (no seed field: the seed asked for, and E312's run of that seed)
    JH.check_head(head(seed=1, pair_run="e312_p_s1"), "e312_p_s1", 1, False)
    JH.check_head(kp350, "e324_s_s0", 0, False)                                                           # E350's head: pair run recorded, no ends = kp
    JH.check_head(att, "e324_s_s0", 0, True)
    JH.check_head(head(seed=0, pair_run="e324_s_s0", ends="kp"), "e324_s_s0", 0, False)
    JH.check_head(NS(), "e312_p_s0", 0, False)                                                            # a Wire without a meta attribute at all
    assert "trained on 'kp' ends" in refused(JH.check_head, legacy, "e324_s_s0", 0, True) and "trained on 'kp' ends" in refused(JH.check_head, kp350, "e324_s_s0", 0, True)       # a kp head on the attached read-out
    assert "trained on 'att' ends" in refused(JH.check_head, att, "e324_s_s0", 0, False), "an att head on the keypoint read-out"
    assert "pair run" in refused(JH.check_head, legacy, "e324_s_s0", 0, False), "E325's head belongs to E312's run"
    assert "pair run" in refused(JH.check_head, att, "e324_s_s1", 0, True) and "pair run" in refused(JH.check_head, head(seed=0, pair_run="e312_p_s0", ends="att"), "e324_s_s0", 0, True), "another pair run"
    assert "seed" in refused(JH.check_head, att, "e324_s_s0", 1, True) and "seed" in refused(JH.check_head, head(seed=1, pair_run="e324_s_s1", ends="att"), "e324_s_s0", 0, True), "another seed"
    assert refused(JH.check_head, head(seed=0, pair_run="e324_s_s0", ends="attached"), "e324_s_s0", 0, True) and refused(JH.check_head, head(seed=0, pair_run="e324_s_s0", ends=None), "e324_s_s0", 0, True), "an unknown kind is not kp"
    assert refused(JH.check_head, head(seed=0, pair_run=None), "e324_s_s0", 0, False), "no pair_run field = E312's run, not the run asked for"


def test_cmd_score_refuses_before_anything_long():
    """through the real `cmd_score` (the temp ROOT has the run folder and the heads, nothing else): a refusal is an AssertionError, a head that passes goes on to the checkpoint of the run and fails there on the missing file"""
    wa = lambda **m: dict(state=JH.Wire().state_dict(), args=dict(m))  # noqa: E731
    heads = dict(kp350=wa(seed=0, pair_run="e324_s_s0", ends="kp"), legacy=wa(seed=0), att=wa(seed=0, pair_run="e324_s_s0", ends="att", head_seed=0), att_wn=wa(seed=0, pair_run="e324_s_s0", ends="att", head_seed="wn", no_jitter=True),
                 att_s1=wa(seed=1, pair_run="e324_s_s1", ends="att"), att_other=wa(seed=0, pair_run="e324_s_s1", ends="att"))
    opts = dict(run="e324_s_s0", seed=0, thr=0.1, full=None, base=None, spec=None, chord=False, snap=True, snap_all=False, snap_guard=False, snap_dedup=False, snap_both=False, snap_nogeom=True, attach_r=None, skip_base_check=True)
    with tempfile.TemporaryDirectory() as d, project_root(d):
        os.makedirs(os.path.join(d, "runs", "e324_s_s0"))
        for name, ck in heads.items():
            os.makedirs(os.path.join(d, "runs", f"h_{name}"))
            torch.save(ck, os.path.join(d, "runs", f"h_{name}", "model.pt"))
        torch.save(heads["att_wn"], os.path.join(d, "runs", "h_att", "model_wn.pt"))

        def score(head, attach, wire="head", seed=0, run="e324_s_s0", tag="t", **more):
            try:
                QE.cmd_score(NS(**dict(opts, seed=seed, run=run, tag=tag, wire=wire, wire_run=head, attach=attach, **more)))
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
        assert "--snap-nogeom" in score("h_att", True, snap_nogeom=False), "an attached read-out without --snap-nogeom is refused before the head is read"
        assert score("h_kp350", False, snap_nogeom=False) == "passed: model.pt", "the keypoint read-out does not need it"
        assert "seed" in score("h_att", True, seed=1) and score(None, True, wire="twin") == "passed: model.pt", "the twin has no head to check"


def test_scorer_command_line_takes_the_options():
    """the staged scorer's argparse takes `--attach --snap --snap-nogeom --wire head` (a run that does not exist fails at its checkpoint, not at the options)"""
    cmd = [sys.executable, os.path.join(STAGE, "jends_eval.py"), "score", "--run", "no_such_run_e358", "--seed", "0", "--thr", "0.1", "--attach", "--snap", "--snap-nogeom", "--wire", "head", "--wire-run", "no_such_head", "--tag", "t"]
    r = subprocess.run(cmd, capture_output=True, text=True, env=dict(os.environ))
    assert r.returncode == 1 and "FileNotFoundError" in r.stderr and "unrecognized" not in r.stderr, (r.returncode, r.stderr[-400:])


def test_staged_selftests_next_to_the_tree():
    QE.selftest()                                                                                         # includes the option checks of `check_wire`
    JH._selftest()
    import jwire as JW                                                                                    # the tree's module under the staged scorer: its selftest (wire_choice with snap_nogeom, attached pairs)
    JW._selftest()


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok", t.__name__)
    print(f"{len(tests)} staged tests ok")
