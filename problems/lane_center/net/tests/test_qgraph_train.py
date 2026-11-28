"""Spec test of stage 0 (`training/train/qgraph_train.py`, its gates `qgates.py`, the steps stage0 and tokens of `lane_train.py`) on real TRAIN / VAL frames,
CPU (forced: CUDA_VISIBLE_DEVICES is emptied before torch is imported), a small B' (width 16), a few steps; everything is written below a temporary run root
(BP.RUNS of this process), nothing under the real one.

    python -P -m lane_center.net.tests.test_qgraph_train          (with the module path: README.md; `research tests lane_center` runs it)

1. phases: every phase (1 - 7, 9) on two TRAIN frames gives a finite loss and a gradient on every tensor of the modules of v1's phase table (V1_TRAIN, a
   literal) and on no other; the frozen stages are v1's (V1_FROZEN).
2. order: `StepBatches` from step s are the batches of the order from step 0 at s (epochs included).
3. stage 0: the three parts of the base config (steps cut to 1 - 2, width 16) through `lane_train stage0`; a stop request written between two parts is obeyed
   by the next (exit 75); a planted failed gate (node precision 1.01 in E292's phase 3) ends the stage with exit GATE_EXIT and no model, the step's log.jsonl
   holds every phase keyed <part>.p<phase>; with the gate off the stage resumes (finished phases loaded, not trained again; a stored gate that could not be
   read is read again) and writes <run>/stage0/model.pt in the format of runs/e292_a_s<S> (state + args + steps), which `lane_net.load_b` loads strictly into
   a `LaneNet`; a finished part with another recipe is refused.
4. resume: phase 6 (it reads the simulated map) stopped by a CONTROL file in the step's folder (exit 75) and started again continues at the step it stopped
   with the same frames and maps, and ends with the weights of the same phase run without a stop, bit for bit.
5. tokens: the cache of that B' on four VAL frames (`lane_tokcache.build` into <run>/tokens) is read by `lane_ptok_data` through the run's ident, whose B' is
   the run's stage 0 (`lane_ptok_data.b_run`, `lane_dump.base_b`)."""
import argparse
import itertools
import json
import os
import tempfile

# the CPU, also where a GPU is visible: the bit-exact resume holds on the CPU (GPU kernels may sum in another order)
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch  # noqa: E402

from trainkit import control
from lane_center.evaluation import lane_dump as DU
from lane_center.net.loss import loss_qgraph as LQG
from lane_center.net.model import jtok as JT
from lane_center.net.model import lane_net as LN
from lane_center.net.model import qgraph as QG
from lane_center.net.prep import grid_data as GD
from lane_center.net.prep import lane_gt_data as LD
from lane_center.net.prep import lane_ptok_data as LP
from lane_center.net.prep import lane_tokcache as LC
from lane_center.training import bflc_paths as BP
from lane_center.training.train import lane_train as LT
from lane_center.training.train import qgates as QGt
from lane_center.training.train import qgraph_train as QT

BASE = json.load(open(os.path.join(BP.ROOT, "training", "configs", "base.json"), encoding="utf-8"))
SMALL = dict(gate_n=2, fit_n=4, workers=0)                                  # the VAL frames of a gate, the TRAIN frames of the E271 threshold, loader workers
GRID = ("stem", "e1", "e2", "e3", "d2", "d1", "d0", "out")                  # the modules of the E271 grid head
# v1's phase table (bev_fusion_lane_center/net/model/qgraph.py of the tag v1): the modules a phase trains, the stages it runs frozen
V1_TRAIN = {1: GRID, 2: ("pyr",), 3: ("nodes",), 4: ("links",), 5: ("nodes",), 6: ("menc", "asg"), 7: ("pyr", "nodes", "links", "menc", "asg"),
            9: ("nodes", "links", "menc", "asg")}
V1_FROZEN = {2: (), 3: ("pyr",), 4: ("pyr", "nodes"), 5: ("pyr", "links"), 6: ("pyr", "nodes", "links"), 7: (), 9: ("pyr",)}


def expect_exit(code, fn):
    try:
        fn()
    except SystemExit as e:
        assert e.code == code, f"exit {e.code}, want {code}"
        return
    raise AssertionError(f"no exit {code}")


def test_phases():
    torch.manual_seed(0)
    gb = GD.collate([GD.GridData("train", ks=[k])[0] for k in (0, 1)])
    ds = LD.QData("train", ks=list(range(4)), map_mode="train")
    qb = LD.collate([ds[0], ds[1]])
    assert QG.PHASE_FROZEN == V1_FROZEN, f"the frozen stages of the phases are not v1's: {QG.PHASE_FROZEN}"
    grid, b = QT.build(QT.parse_args(["--name", "x", "--phases", "1", "--width", "16"]))[0], LN.build_b(dict(width=16, soft_nms=["mnms", 1.75, 0.1]))
    for k in (1, 2, 3, 4, 5, 6, 7, 9):
        m, batch = (grid, gb) if k == 1 else (b, qb)
        m.zero_grad(set_to_none=True)
        QT.phase_params(m, k)
        loss, info = QT.phase_loss(k, "cpu")(m, batch)
        assert torch.isfinite(loss), (k, info)
        loss.backward()
        got = {n for n, p in m.named_parameters() if p.grad is not None and torch.isfinite(p.grad).all()}
        want = {n for n, _ in m.named_parameters() if n.split(".")[0] in V1_TRAIN[k]}
        assert got == want, f"phase {k}: the modules of v1's table {V1_TRAIN[k]}: no gradient on {sorted(want - got)[:5]}, one outside them on {sorted(got - want)[:5]}"
        print(f"phase {k}: loss {float(loss):.3f}, a gradient on every tensor of {V1_TRAIN[k]} ({len(want)}) and no other")


def test_order():
    full = list(itertools.islice(QT.StepBatches(10, 3, 7), 9))
    assert all(len(set(sum(full[e * 3:(e + 1) * 3], []))) == 9 for e in range(3)) and full[0] != full[3], "an epoch is three batches without repeats, a new order each epoch"
    for s in (1, 3, 5):
        assert list(itertools.islice(QT.StepBatches(10, 3, 7, start=s), 4)) == full[s:s + 4], s


def small_cfg(name, e292_gate):
    """the base config with stage 0 on, width 16, 1 - 2 steps per phase, no loader workers"""
    s0 = {p: dict(BASE["stage0"][p], width=16, workers=0, bs=2, steps=[2 if p == "e271" else 1] * len(BASE["stage0"][p]["phases"])) for p in LT.STAGE0}
    s0["e292"].update(e292_gate)
    return dict(BASE, name=name, seed=0, out_root=BP.RUNS, stage0=dict(s0, enabled=1), tokens=dict(BASE["tokens"], enabled=1, workers=0))


def run_stage0(tmp, cfg):
    p = os.path.join(tmp, "cfg.json")
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh)
    return LT.main(["stage0", p])


def test_stage0(tmp):
    keep = QT.main
    trained, stop_after = [], [None]

    def small_main(argv, step_dir):
        a = QT.parse_args(argv)
        trained.append(a.name)
        out = QT.train(a, BP.RUNS, step_dir, **SMALL)
        if a.name.endswith(f"/{stop_after[0]}"):
            # as the worker writes it while a part ends (its gate, the next part's build)
            control.request_stop(step_dir, "test: between two parts")
        return out
    try:
        QT.main = small_main
        stop_after[0] = "e271"                                                                # a stop between two parts is obeyed by the next part
        expect_exit(control.EXIT_STOPPED, lambda: run_stage0(tmp, small_cfg("E1.stop_s0", dict(gate_phases=[]))))
        assert trained == ["E1.stop_s0/stage0/e271", "E1.stop_s0/stage0/e273"] and not os.path.exists(os.path.join(BP.RUNS, "E1.stop_s0", "stage0", "e273", "p2", "ckpt.pt"))
        assert os.path.exists(os.path.join(BP.RUNS, "E1.stop_s0", "stage0", "STOPPED")), "the stop between two parts was not obeyed"
        stop_after[0], trained[:] = None, []
        expect_exit(QT.GATE_EXIT, lambda: run_stage0(tmp, small_cfg("E1.s0_s0", dict(gate_phases=[3], gate_node_prec=1.01))))
        run = os.path.join(BP.RUNS, "E1.s0_s0", "stage0")
        assert not os.path.exists(os.path.join(run, "model.pt")) and not os.path.exists(os.path.join(run, "e292", "model.pt")), "a failed gate wrote a model"
        status = os.path.join(run, "e292", "p3", "status.json")
        assert "node precision within 2 m" in json.load(open(status))["gate"]["why"]
        rows = [json.loads(line) for line in open(os.path.join(run, "log.jsonl"))]
        assert {(r["part"], r["phase"]) for r in rows} == {("e271", 1)} | {("e273", k) for k in range(2, 8)} | {("e292", 3)} and any(r.get("e292.p3.gate_pass") == 0.0 for r in rows)
        assert all(isinstance(r["e273.p7.total"], float) for r in rows if (r["part"], r["phase"]) == ("e273", 7) and "e273.p7.gate_pass" not in r), "a row's numbers are keyed <part>.p<phase>"
        st = json.load(open(status))
        st["gate"] = {"pass": False, "why": QGt.UNREAD + ": planted (an OOM in the gate)"}
        json.dump(st, open(status, "w"))
        mtime = os.path.getmtime(os.path.join(run, "e273", "p7", "model.pt"))
        assert run_stage0(tmp, small_cfg("E1.s0_s0", dict(gate_phases=[]))) == 0
        assert trained == ["E1.s0_s0/stage0/" + p for p in LT.STAGE0] + ["E1.s0_s0/stage0/e292"], trained
        assert os.path.getmtime(os.path.join(run, "e273", "p7", "model.pt")) == mtime, "a finished part was trained again"
        assert not json.load(open(status))["gate"]["why"].startswith(QGt.UNREAD), "a gate that could not be read was not read again on the restart"
        refused = small_cfg("E1.s0_s0", dict(gate_phases=[]))
        refused["stage0"]["e292"]["bs"] = 3
        try:
            QT.train(QT.parse_args(LT.stage0_args(refused, "e292")), BP.RUNS, **SMALL)
            raise RuntimeError("a finished part with another recipe was accepted")
        except AssertionError as e:
            assert "another recipe" in str(e), e
    finally:
        QT.main = keep
    ck = LN.load_checkpoint(os.path.join(run, "model.pt"))
    assert sorted(ck) == ["args", "state", "steps"] and ck["steps"] == {3: 1, 4: 1, 5: 1, 6: 1, 9: 1} and ck["args"]["soft_nms"] == ["mnms", 1.75, 0.1]
    net = LN.LaneNet(LN.load_b(run), K=512)                                                   # load_b: strict
    assert net.b.mnms == (1.75, 0.1) and set(net.b.state_dict()) == set(ck["state"])
    with open(os.path.join(run, "e292", "model.pt"), "rb") as a, open(os.path.join(run, "model.pt"), "rb") as b:
        assert a.read() == b.read(), "stage0/model.pt is the last part's, byte for byte"
    print("stage 0: three parts, a failed gate exits", QT.GATE_EXIT, "and the resumed stage writes a model.pt that LaneNet loads strictly")
    return run


def test_resume(tmp, head):
    rows, calls = {}, [0]
    loss = LQG.qgraph_loss

    def recording(stop_dir=None, at=None):
        def fn(m, b, k):
            calls[0] += 1
            rows.setdefault(stop_dir, []).append(b["row"].tolist())
            if calls[0] == at:
                control.request_stop(stop_dir, "test")
            return loss(m, b, k)
        return fn

    argv = lambda name: ["--name", name, "--phases", "6", "--steps", "4", "--bs", "2", "--width", "16", "--workers", "0", "--head", head, "--gate-phases"]  # noqa: E731
    try:
        LQG.qgraph_loss = recording()
        QT.train(QT.parse_args(argv("R.whole")), BP.RUNS, **SMALL)
        stop = os.path.join(BP.RUNS, "R.cut")
        os.makedirs(stop)
        calls[0] = 0
        LQG.qgraph_loss = recording(stop, at=2)
        expect_exit(control.EXIT_STOPPED, lambda: QT.train(QT.parse_args(argv("R.cut")), BP.RUNS, **SMALL))
        assert json.load(open(os.path.join(stop, "STOPPED")))["step"] == 2 and not os.path.exists(os.path.join(stop, "p6", ".done"))
        LQG.qgraph_loss = recording(stop, at=-1)
        QT.train(QT.parse_args(argv("R.cut")), BP.RUNS, **SMALL)
    finally:
        LQG.qgraph_loss = loss
    assert rows[stop] == rows[None], f"the frames of the stopped and resumed phase {rows[stop]} are not those of the whole one {rows[None]}"
    assert "resumed from" in open(os.path.join(stop, "p6", "log.txt")).read() and "at step 2" in open(os.path.join(stop, "p6", "log.txt")).read()
    a, b = (LN.load_checkpoint(os.path.join(BP.RUNS, n, "model.pt"))["state"] for n in ("R.whole", "R.cut"))
    assert all(torch.equal(a[k], b[k]) for k in a), "the resumed phase does not end with the weights of the whole one"
    print("resume: stopped at step 2 (exit 75), resumed with the same frames, the same weights bit for bit")


def test_tokens(run):
    a = argparse.Namespace(run="runs/E1.s0_s0/stage0", split="val", K=512, f2=True, limit=4, workers=0, out="runs/E1.s0_s0/tokens")
    LC.build(a, lambda r, dev: LN.LaneNet(LN.load_b(r, dev), K=a.K), JT.pair_geometry)
    ident = LT.own_tokens(dict(name="E1.s0_s0"))
    ds = LP.PTokData("val", 0, ident)
    meta = json.load(open(os.path.join(ds.dir, "meta.json")))
    assert len(ds) == 4 and ds.K == 512 and ds.dir == os.path.join(BP.RUNS, "E1.s0_s0", "tokens", "val") and meta["run_sha"] == BP.file_sha256(os.path.join(run, "model.pt"))[:16]
    b = LP.collate([ds[0], ds[1]])
    assert b["tok"].shape == (2, 512, 64) and b["y"].shape == (2, 512, 512)
    assert LP.b_run(ident, 0) == DU.base_b(DU.cache_kw(dict(seed=0, ident=ident))) == "runs/E1.s0_s0/stage0" and LP.b_run("e292_a", 1) == "runs/e292_a_s1"
    print("tokens: the cache of the run's B' on 4 VAL frames, read through the run's ident; its B' is the run's stage 0")


if __name__ == "__main__":
    test_phases()
    test_order()
    keep = BP.RUNS
    with tempfile.TemporaryDirectory() as tmp:
        try:
            BP.RUNS = tmp                                                                         # this test's run root (resolve, lane_train, the trainers)
            stage0 = test_stage0(tmp)
            test_resume(tmp, "runs/E1.s0_s0/stage0/e271/model.pt")
            test_tokens(stage0)
        finally:
            BP.RUNS = keep
    print("test_qgraph_train ok")
