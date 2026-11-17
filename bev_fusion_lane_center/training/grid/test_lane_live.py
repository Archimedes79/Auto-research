"""Tests of the live token producer of the S4 joint fine-tune (`data/lane_live_tok.py`, E372).

    wsl bash tools/env/_py14.sh test_lane_live.py            # the selftest (a toy `QGraph` on the CPU: node targets by hand, the live selection against `select_tokens`, the graph and its gradient at the chosen cells, the labels by hand, the token drop, the loader, the freezing, the
                                                             # gradient check of the card), `check` + `fixture` + `grad` on real frames (the node targets against QData; the label step on the cached tokens exactly the cache; the gradient check with the real B'
                                                             # and the champion head), then the hand mutants (about 10 minutes)
    wsl bash tools/env/_py14.sh test_lane_live.py selftests  # the selftest and the real-data commands only

A mutant = one textual change of the source (the wrong frame, a guard removed, the float16 rounding of tok or its gradient, a graph cut in the gather, e and e_raw swapped, the labels of another K, the node targets transposed, the pyramid in train mode, the node head frozen,
the pyramid trainable, a label made from tokens that carry the graph, the pair gate of the wrong direction, the token drop ignored, ...); the mutated copy runs `selftest` in a temporary folder with BFLC_ROOT pointing at this project, four at a time, and every mutant must FAIL it.
The real-data commands need the E365 caches (VAL and TRAIN of B' seeds 0 and 1) and the node-table cache of E357; the GPU fixture (`lane_live_tok.py fixture --gpu`) is a run of the lane chat, not part of this test.
`import numpy, torch` walks the 319 project directories of PYTHONPATH on /mnt/c first (50 s on this box): the child processes import the two with that path moved behind site-packages, then restore it.
"""
import concurrent.futures as cf
import os
import subprocess
import sys
import tempfile

ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA = os.path.join(ROOT, "training", "data")
MOD = "lane_live_tok.py"
ENV = dict(os.environ, BFLC_ROOT=ROOT, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")                      # a shared box: one thread per test process
LAUNCH = ("import os, sys, runpy; o = list(sys.path); r = os.environ['BFLC_ROOT']; sys.path[:] = [p for p in o if not p.startswith(r)] + [p for p in o if p.startswith(r)]; import numpy, torch; sys.path[:] = o; "
          "p = sys.argv[1]; sys.argv = sys.argv[1:]; runpy.run_path(p, run_name='__main__')")

MUTANTS = [
    ("the item of frame i is frame i + 1", "o = self.td[i]", "o = self.td[(i + 1) % len(self.td)]"),
    ("the TEST guard is removed", "LPD.check_rows(split, self.rows, expect=None)", "pass"),
    ("the order check ignores the rows", "assert [int(r) for r in fr[\"rows\"]] == [int(r) for r in rows] and list(fr[\"tokens\"]) == list(tokens)", "assert list(fr[\"tokens\"]) == list(tokens)"),
    ("the order check ignores the tokens", "assert [int(r) for r in fr[\"rows\"]] == [int(r) for r in rows] and list(fr[\"tokens\"]) == list(tokens)", "assert [int(r) for r in fr[\"rows\"]] == [int(r) for r in rows]"),
    ("tok is not rounded to float16", "return x.half().float()", "return x.clone()"),
    ("the gradient of tok is rounded to float16", "    def backward(ctx, g):\n        return g", "    def backward(ctx, g):\n        return g.half().float()"),
    ("tok is gathered from the wrong cells", "tk[\"cell\"].long()[:, None, :].expand(", "tk[\"cell\"].long().flip(1)[:, None, :].expand("),
    ("the graph is cut in the gather of tok", "tok = no[\"tok\"].flatten(2).gather(", "tok = no[\"tok\"].detach().flatten(2).gather("),
    ("the graph is cut in the token inputs", "batch = {k: tk[k] for k in GRAPH_KEYS}", "batch = {k: tk[k].detach() for k in GRAPH_KEYS}"),
    ("e and e_raw are swapped", "batch = {k: tk[k] for k in GRAPH_KEYS}", "batch = {k: tk[\"e_raw\" if k == \"e\" else \"e\" if k == \"e_raw\" else k] for k in GRAPH_KEYS}"),
    ("the head's input keys lose the flags", "GRAPH_KEYS = tuple(k for k in LP.TOKEN_FIELDS if k != \"valid\")", "GRAPH_KEYS = tuple(k for k in LP.TOKEN_FIELDS if k not in (\"valid\", \"flags\"))"),
    ("the labels are those of another K", "tables[b], K=K, allowed=allowed[b])", "tables[b], K=K // 2, allowed=allowed[b])"),
    ("the labels are made of tokens that carry the graph", "tn = {k: tk[k].detach().cpu().numpy() for k in", "tn = {k: tk[k].cpu().numpy() for k in"),
    ("the pair gate is read from the wrong direction", "tables[b], K=K, allowed=allowed[b])", "tables[b], K=K, allowed=allowed[b].T)"),
    ("ylong is classed by twice the chord", "axis=-1) > QT.SHORT)", "axis=-1) > 2 * QT.SHORT)"),
    ("the row of a frame is that of the first", "row=int(row[b])", "row=int(row[0])"),
    ("the token of a frame is that of the first", "token=token[b], row=", "token=token[0], row="),
    ("the token drop is ignored", "keep = keep_nodes(token[b], K, token_drop, seed)", "keep = np.ones(K, bool)"),
    ("the token drop ignores the seed", "keep = keep_nodes(token[b], K, token_drop, seed)", "keep = keep_nodes(token[b], K, token_drop, 0)"),
    ("a removed token stays valid", "valid = tn[\"valid\"][b] & keep", "valid = tn[\"valid\"][b]"),
    ("a removed token keeps its type", "ytype=np.where(keep, lab[\"gt_type\"], 0).astype(np.int64)", "ytype=lab[\"gt_type\"].astype(np.int64)"),
    ("a removed token keeps its pairs", "K, keep if token_drop > 0 else None)", "K, None)"),
    ("tokens_np keeps the removed tokens valid", "cell=tn[\"cell\"][b].astype(np.int64), valid=valid.copy()),", "cell=tn[\"cell\"][b].astype(np.int64), valid=tn[\"valid\"][b].copy()),"),
    ("the source does not pass the token drop on", "live_batch(no, tables, token, row, self.K, token_drop, seed)", "live_batch(no, tables, token, row, self.K)"),
    ("the node targets are transposed", "return {k: g[k] for k in NODE_KEYS}", "return {k: np.swapaxes(g[k], -1, -2) for k in NODE_KEYS}"),
    ("B's loss runs on tokens without the graph", "return QG.node_loss(no, targets)", "return QG.node_loss({k: v.detach() for k, v in no.items()}, targets)"),
    ("B's loss accepts any loss switch", "assert not bad, f\"`qgraph.CFG` is not that of B''s training", "assert True, f\"`qgraph.CFG` is not that of B''s training"),
    ("the pyramid is left in train mode", "        super().train(mode)\n        self.model.eval()\n", "        super().train(mode)\n"),
    ("the node head is frozen", "                p.requires_grad_(True)", "                p.requires_grad_(False)"),
    ("the pyramid is not frozen", "        for p in self.model.parameters():\n            p.requires_grad_(False)", "        for p in self.model.nodes.parameters():\n            p.requires_grad_(False)"),
    ("train_nodes=False is ignored", "        if train_nodes:\n", "        if True:\n"),
    ("trainable() lists the frozen parameters", "return [p for p in self.parameters() if p.requires_grad]", "return [p for p in self.parameters()]"),
    ("B' runs inside the trainer's autocast", "        with torch.autocast(device_type=dev.type, enabled=False):         # float32", "        if True:         # float32"),
    ("compare_frame aligns by rank, not by cell", "    ja, jw = np.array([ia[c] for c in com], int), np.array([iw[c] for c in com], int)", "    ja = jw = np.arange(len(a[\"cell\"]))"),
    ("compare_frame ignores the pairs y", "for k in (\"y\", \"ylong\")})", "for k in (\"ylong\",)})"),
    ("compare_frame counts a one-sided NaN as equal", "np.where(np.isnan(x) | np.isnan(y), np.inf,", "np.where(np.isnan(x) | np.isnan(y), 0.0,"),
    ("compare_frame ignores the label counts of gt_id", "for k in (\"gt_id\", \"gt_flags\", \"gt_aux\"))", "for k in (\"gt_flags\", \"gt_aux\"))"),
]


def run_selftest():
    r = subprocess.run([sys.executable, "-c", LAUNCH, os.path.join(DATA, MOD), "selftest"], capture_output=True, text=True, env=ENV)
    assert r.returncode == 0, f"{MOD} selftest failed:\n{r.stdout[-800:]}{r.stderr[-2500:]}"
    print(r.stdout.strip().splitlines()[-1], flush=True)


def run_real():
    """the real-data commands: the node targets against QData (VAL and TRAIN, 20 frames each), the fixture on 8 frames of both seeds (exit code 0 = the label step on the cached tokens is exactly the cache) and the gradient check of the card on 2 real frames with the
    champion head (the head's loss reaches every node-head parameter of the real B', none of the frozen stages, the pyramid is unchanged by a step)"""
    for args in (["check", "--split", "val", "--n", "20"], ["check", "--split", "train", "--n", "20"], ["fixture", "--n", "8", "--seeds", "0", "1"], ["grad", "--n", "2", "--head", "e365_s1_s0"]):
        r = subprocess.run([sys.executable, "-c", LAUNCH, os.path.join(DATA, MOD)] + args, capture_output=True, text=True, env=ENV)
        assert r.returncode == 0, f"{MOD} {' '.join(args)} failed:\n{r.stdout[-1500:]}{r.stderr[-1500:]}"
        print(f"{' '.join(args)}: {r.stdout.strip().splitlines()[-1]}", flush=True)


def one(args):
    k, (name, a, b), tmp = args
    src = open(os.path.join(DATA, MOD), encoding="utf-8").read()
    if src.count(a) != 1:
        return name, f"NOT APPLIED ({src.count(a)} matches)"
    d = os.path.join(tmp, str(k))
    os.makedirs(d)
    p = os.path.join(d, MOD)
    with open(p, "w", encoding="utf-8") as fh:
        fh.write(src.replace(a, b))
    r = subprocess.run([sys.executable, "-c", LAUNCH, p, "selftest"], capture_output=True, text=True, env=ENV)
    return name, "killed" if r.returncode != 0 else "SURVIVED"


def run_mutants(workers=4):
    left = []
    with tempfile.TemporaryDirectory() as tmp, cf.ThreadPoolExecutor(workers) as ex:
        for name, res in ex.map(one, [(k, m, tmp) for k, m in enumerate(MUTANTS)]):
            print(f"{res:<12} {name}", flush=True)
            if res != "killed":
                left.append(name)
    print(f"{len(MUTANTS) - len(left)} of {len(MUTANTS)} killed", flush=True)
    return left


if __name__ == "__main__":
    run_selftest()
    run_real()
    if len(sys.argv) < 2 or sys.argv[1] != "selftests":
        left = run_mutants()
        assert not left, f"mutants survived: {left}"
    print("test_lane_live ok")
