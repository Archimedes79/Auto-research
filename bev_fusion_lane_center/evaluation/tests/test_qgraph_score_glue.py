"""Test of the glue of `qgraph_run_eval.score` (E275 read-outs) on a toy lane, CPU, no model: `predict_val`, `chain_vertices` and `val_frames` are replaced by toy data and ROOT by a temporary folder, so
a typo in the part that only runs on the GPU costs nothing. Checks: the old call (readouts=None) writes one card in the old place with the old keys; the read-out call writes one card per spec
in `runs/<exp>/<run>.<spec>/` with the model as `run` and the step's counts; plain@0.5 equals the old card; the layer-1 comparison with a stored card; the kill criterion of the judged spec stops
before any card is written and only warns for the 0.4 spec; the reproduction gate against an E274 sweep file; the radius curve; the dump; `--readouts` without `--exp` is refused.

    wsl bash tools/env/_py14.sh test_qgraph_score_glue.py
"""
import contextlib
import io
import json
import os
import sys
import tempfile
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for sub in ("..", "../../training/grid", "."):
    sys.path.insert(0, os.path.normpath(os.path.join(HERE, sub)))
import qformat  # noqa: E402
import qgraph_eval as QE  # noqa: E402
import qgraph_run_eval as QR  # noqa: E402
import test_qgraph_readout as T  # noqa: E402

N = 6


def setup():
    g, old = QE.toy_lane_gt()
    base = qformat.oracle_prediction(g)

    class F(QE.Frame):
        def lane(self):
            return g

        def ref(self):
            if self._ref is None:
                self._ref = QE.LaneRef(g)
            return self._ref
    frames = [F(k, k, f"tok{k}", k // 2, old) for k in range(N)]
    QE.val_frames = lambda: frames
    QE.stored_share = lambda fr: 1.0

    def predict_val(run, mode, n=0, device="cpu"):
        ds = types.SimpleNamespace(bd=types.SimpleNamespace(rows=list(range(N))), ks=list(range(N)))
        out = []
        for k in range(N):
            p = T.with_duplicate(base, 2 + k, [0.4, 0.0], 0.6)                                           # a duplicate node next to a different oracle node in every frame
            out.append(T.with_duplicate(p, 10, [6.0, 0.0], 0.45, link=False))                           # a false node far from every lane at e = 0.45: delivered at 0.4, not at 0.5
        return ds, out
    QR.predict_val = predict_val

    def chain_vertices(seed, n=0, device="cpu"):
        V = np.stack([np.zeros(41), np.linspace(-40, 40, 41)], 1)
        H = np.tile([0.0, 1.0], (41, 1))
        return [(V, H, np.ones(41)) for _ in range(N)]
    QR.chain_vertices = chain_vertices


def quiet(f, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return f(*a, **k)


def main():
    keep = (QE.val_frames, QE.stored_share, QR.predict_val, QR.chain_vertices, QR.ROOT)
    try:
        setup()
        with tempfile.TemporaryDirectory() as root:
            QR.ROOT = root
            R = os.path.join(root, "runs")
            ro = ["plain@0.5", "close:1.5@0.5", "close:1.5@0.4", "plain@0.4"]
            # the old call: one card in the old place, the old keys, no read-out key
            c0 = quiet(QR.score, "toy", 0, 0, "cpu", exp="e273")
            assert os.listdir(os.path.join(R, "e273")) == ["toy"] and list(c0) == ["run", "seed", "split", "frames", "date", "d_ref", "p", "g", "diff", "seconds"], list(c0)
            old = json.load(open(os.path.join(R, "e273", "toy", "card.json")))
            # the read-out call
            dump = os.path.join(root, "temp", "e275", "pred_toy.pkl")
            cards = quiet(QR.score, "toy", 0, 0, "cpu", exp="e275", readouts=ro, curve=[1.0, 2.0], dump=dump)
            assert sorted(os.listdir(os.path.join(R, "e275"))) == sorted([f"toy.{n}" for n in ("plain_t0.5", "close1.5_t0.5", "close1.5_t0.4", "plain_t0.4")] + ["toy.curve.json"])
            assert [s.kind for s in cards] == ["plain", "close", "close", "plain"]
            p5 = json.load(open(os.path.join(R, "e275", "toy.plain_t0.5", "card.json")))
            a = {k: v for k, v in old.items() if k not in ("date", "seconds")}
            b = {k: v for k, v in p5.items() if k not in ("date", "seconds", "readout")}
            assert json.dumps(a, sort_keys=True, default=float) == json.dumps(b, sort_keys=True, default=float)               # plain@0.5 is the old card
            assert p5["readout"]["reproduces_stored_card_max_abs_diff"] == 0.0                                              # layer 1: the stored card at the same frames
            c5 = json.load(open(os.path.join(R, "e275", "toy.close1.5_t0.5", "card.json")))
            r = c5["readout"]
            assert c5["run"] == "toy" and r["spec"] == "close1.5_t0.5" and r["radius"] == 1.5 and r["thr_e"] == 0.5 and r["nodes_after"] < r["nodes_before"] and "tied_pairs_within_radius" in r and r["ms_per_frame"] > 0
            assert abs(r["dropped_frac"] - (1 - r["nodes_after"] / r["nodes_before"])) < 1e-12
            assert c5["p"]["nodes"]["node_P"] > p5["p"]["nodes"]["node_P"]                                                  # the duplicates are gone: precision up
            assert c5["g"] == p5["g"] and "lane_acc" in c5["p"]["assign_exact"] and "lane_acc" in c5["p"]["assign_error"]    # g is shared (the toy duplicates carry the assignment of their parent)
            c4 = json.load(open(os.path.join(R, "e275", "toy.plain_t0.4", "card.json")))
            assert "nodes_before" not in c4["readout"] and c4["readout"]["thr_e"] == 0.4                                    # no step, no counts
            assert c4["p"]["nodes"]["node_P"] < p5["p"]["nodes"]["node_P"]                                                  # the threshold reaches the scorer: the node at e = 0.45 is delivered at 0.4 only
            c54 = json.load(open(os.path.join(R, "e275", "toy.close1.5_t0.4", "card.json")))
            assert c54["readout"]["nodes_before"] > r["nodes_before"] and c54["p"]["nodes"]["node_P"] < c5["p"]["nodes"]["node_P"]   # ... also behind the step
            assert "# toy [close1.5_t0.5]" in open(os.path.join(R, "e275", "toy.close1.5_t0.5", "results.md")).read().split("\n")[0]
            cv = json.load(open(os.path.join(R, "e275", "toy.curve.json")))
            assert sorted(cv["curve"]) == ["1", "2"] and cv["frames"] == N and cv["curve"]["2"]["node_R"] <= cv["curve"]["1"]["node_R"]
            import pickle
            d = pickle.load(open(dump, "rb"))
            assert len(d) == N and [x["row"] for x in d] == list(range(N))
            # the kill criterion: the judged spec (threshold 0.5) stops before any card is written ...
            try:
                quiet(QR.score, "toy_kill", 0, 0, "cpu", exp="e275", readouts=["close:1.5@0.5"], max_drop=0.001)
                raise SystemExit("the kill criterion did not stop the scoring")
            except SystemExit as e:
                assert "KILL CRITERION" in str(e), e
            assert not [x for x in os.listdir(os.path.join(R, "e275")) if x.startswith("toy_kill")]
            # ... a reported spec (threshold 0.4) only warns
            quiet(QR.score, "toy_warn", 0, 0, "cpu", exp="e275", readouts=["close:1.5@0.4"], max_drop=0.001)
            assert os.path.isdir(os.path.join(R, "e275", "toy_warn.close1.5_t0.4"))
            # the reproduction gate against an E274 sweep file
            g5 = c5["p"]["graph"]
            sweep = {"pos1.5_e0.5": dict(topo_F1=g5["topo_F1"], F1_1=g5["F1@1.0"], junction_F1=g5["junction_F1"], lanes_found=g5["lanes_found"], node_P=c5["p"]["nodes"]["node_P"], node_R=c5["p"]["nodes"]["node_R"])}
            ok, bad = os.path.join(root, "sweep_ok.json"), os.path.join(root, "sweep_bad.json")
            json.dump(sweep, open(ok, "w"))
            json.dump({"pos1.5_e0.5": {**sweep["pos1.5_e0.5"], "topo_F1": g5["topo_F1"] + 0.02}}, open(bad, "w"))
            quiet(QR.score, "toy_ok", 0, 0, "cpu", exp="e275", readouts=["close:1.5@0.5"], repro=ok)
            assert json.load(open(os.path.join(R, "e275", "toy_ok.close1.5_t0.5", "card.json")))["readout"]["reproduces_e274_sweep_max_abs_diff"] < 1e-12
            try:
                quiet(QR.score, "toy_bad", 0, 0, "cpu", exp="e275", readouts=["close:1.5@0.5"], repro=bad)
                raise SystemExit("the reproduction gate did not stop the scoring")
            except SystemExit as e:
                assert "does not reproduce" in str(e), e
            assert not [x for x in os.listdir(os.path.join(R, "e275")) if x.startswith("toy_bad")]
            # --readouts without --exp is refused (the E273 cards must not receive read-out cards)
            sys.argv = ["qgraph_run_eval.py", "score", "--run", "toy", "--readouts", "plain@0.5"]
            try:
                with contextlib.redirect_stderr(io.StringIO()):
                    QR.main()
                raise SystemExit("--readouts without --exp accepted")
            except SystemExit as e:
                assert e.code == 2, e.code
    finally:
        QE.val_frames, QE.stored_share, QR.predict_val, QR.chain_vertices, QR.ROOT = keep
    print("qgraph_run_eval.score glue (toy lane, CPU): old call unchanged, one card per read-out in its own folder, plain@0.5 = old card, kill criterion (judged spec stops, 0.4 spec warns), "
          "reproduction gate, curve, dump, --exp required with --readouts: ok")


if __name__ == "__main__":
    main()
