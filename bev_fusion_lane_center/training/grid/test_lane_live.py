"""Tests of the live token producer of the S4 joint fine-tune (`data/lane_live_tok.py`, E372).

    wsl bash tools/env/_py14.sh test_lane_live.py            # the selftest (a toy `QGraph` on the CPU: node targets by hand, the live selection against `select_tokens`, the graph and its gradient at the chosen cells, the labels and the values of tokens_np by hand, the gate,
                                                             # the PMAX overflow, the token drop, the loader and its guards, `compare_frame` / `_vs_cache`, the freezing, tokens + labelled = forward, the head reads only its inputs, the gradient check of the card),
                                                             # `check` + `fixture` + `grad` on real frames (the node targets against QData and the cache's checkpoint; the label step on the cached tokens exactly the cache at bs 4 and 8; the gradient check with the real B'
                                                             # and the champion head), then the hand mutants (about 15 minutes)
    wsl bash tools/env/_py14.sh test_lane_live.py selftests  # the selftest and the real-data commands only

A mutant = one textual change of the source (the wrong frame or table, a guard removed, the float16 rounding of tok or its gradient, a graph cut in the gather, e and e_raw swapped, a field of tokens_np taken from another, the labels of another K, the node targets
transposed, the pyramid in train mode, the node head frozen, a label made from tokens that carry the graph, the pair gate ignored or of the wrong direction, the overflow not named, the token drop ignored, a fixture that compares nothing, ...); the mutated copy runs
`selftest` in a temporary folder with BFLC_ROOT pointing at this project, four at a time. A mutant counts as KILLED only when the child's own output shows an AssertionError (or `refuses`' "accepted although it must be refused"), or, for a mutant that is killed by a crash
on purpose, the error text named in its 4th field; any other non-zero exit is reported as CRASHED (a NameError or KeyError is a bug of the mutant or of the test, not a detection) and fails the test, like a mutant that survives.
The real-data commands need the E365 caches (VAL and TRAIN of B' seeds 0 and 1) and the node-table cache of E357; the GPU fixture (`lane_live_tok.py fixture --gpu`) is a run of the lane chat, not part of this test.
`import numpy, torch` walks the 319 project directories of PYTHONPATH on /mnt/c first (50 s on this box): the child processes import the two with that path moved behind site-packages, then restore it.
"""
import concurrent.futures as cf
import os
import re
import subprocess
import sys
import tempfile

ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA = os.path.join(ROOT, "training", "data")
MOD = "lane_live_tok.py"
ENV = dict(os.environ, BFLC_ROOT=ROOT, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")                      # a shared box: one thread per test process
LAUNCH = ("import os, sys, runpy; o = list(sys.path); r = os.environ['BFLC_ROOT']; sys.path[:] = [p for p in o if not p.startswith(r)] + [p for p in o if p.startswith(r)]; import numpy, torch; sys.path[:] = o; "
          "p = sys.argv[1]; sys.argv = sys.argv[1:]; runpy.run_path(p, run_name='__main__')")
ASSERTED = re.compile(r"^(AssertionError|RuntimeError: accepted although)", re.M)                       # how a test names its own failure

HEAD_IN = "return {k: tk[k] for k in GRAPH_KEYS + (\"valid\",)}"
LABEL_CALL = "tables[b], K=K, allowed=None if allowed is None else allowed[b])"
FWD = "return dict(no=no, batch=self.labelled(tk, tables, token, row, token_drop, seed, gate))"
LAB_CALL = "live_batch(tk, tables, token, row, self.K, token_drop, seed, gate)"
TNP = "floats.update({\"np_\" + k: gap(tok(a[\"tnp\"], k, ja), tok(w[\"tnp\"], k, jw)) for k in NP_KEYS})"
CHECK_ORDER = "check_order(cache_dir or LP.cache_dir(split, seed, ident), self.rows, self.tokens)"
ORDER_ASSERT = "assert [int(r) for r in fr[\"rows\"]] == [int(r) for r in rows] and list(fr[\"tokens\"]) == list(tokens)"

MUTANTS = [
    ("the item of frame i is frame i + 1", "o = self.td[i]", "o = self.td[(i + 1) % len(self.rows)]"),
    ("every frame of a batch gets the table of the first", "table=[o[\"table\"] for o in b]", "table=[b[0][\"table\"] for o in b]"),
    ("the TEST guard is removed", "LPD.check_rows(split, self.rows, expect=None)", "pass"),
    ("LiveData skips the order check", CHECK_ORDER, "pass"),
    ("the order check ignores the rows", ORDER_ASSERT, "assert list(fr[\"tokens\"]) == list(tokens)"),
    ("the order check ignores the tokens", ORDER_ASSERT, "assert [int(r) for r in fr[\"rows\"]] == [int(r) for r in rows]"),
    ("the checkpoint of the cache is not checked", "assert got == want, f\"{run}/model.pt", "assert True, f\"{run}/model.pt"),
    ("tok is not rounded to float16", "return x.half().float()", "return x.clone()"),
    ("the gradient of tok is rounded to float16", "    def backward(ctx, g):\n        return g", "    def backward(ctx, g):\n        return g.half().float()"),
    ("tok is gathered from the wrong cells", "tk[\"cell\"].long()[:, None, :].expand(", "tk[\"cell\"].long().flip(1)[:, None, :].expand("),
    ("the graph is cut in the gather of tok", "tok = no[\"tok\"].flatten(2).gather(", "tok = no[\"tok\"].detach().flatten(2).gather("),
    ("the graph is cut in the token inputs", HEAD_IN, "return {k: tk[k].detach() for k in GRAPH_KEYS + (\"valid\",)}"),
    ("e and e_raw are swapped", HEAD_IN, "return {k: tk[\"e_raw\" if k == \"e\" else \"e\" if k == \"e_raw\" else k] for k in GRAPH_KEYS + (\"valid\",)}"),
    ("the head's inputs lack valid", HEAD_IN, "return {k: tk[k] for k in GRAPH_KEYS}"),
    ("the head's input keys lose the flags", "GRAPH_KEYS = tuple(k for k in LP.TOKEN_FIELDS if k != \"valid\")", "GRAPH_KEYS = tuple(k for k in LP.TOKEN_FIELDS if k not in (\"valid\", \"flags\"))"),
    ("the labels are those of another K", LABEL_CALL, "tables[b], K=K // 2, allowed=None if allowed is None else allowed[b])", "could not be broadcast"),
    ("the labels are made of tokens that carry the graph", "tn = {k: tk[k].detach().cpu().numpy() for k in", "tn = {k: tk[k].cpu().numpy() for k in", "requires grad"),
    ("the pair gate is read from the wrong direction", "else allowed[b])", "else allowed[b].T)"),
    ("the pair gate is computed although it is off", "    allowed = None\n    if gate:\n", "    allowed = None\n    if True:\n"),
    ("a frame over PMAX is not named", "raise LabelOverflow(token[b], int(m.group(1))) from e", "raise"),
    ("a PMAX overflow names the wrong frame", "raise LabelOverflow(token[b], int(m.group(1)))", "raise LabelOverflow(token[0], int(m.group(1)))"),
    ("a PMAX overflow names the wrong count", "raise LabelOverflow(token[b], int(m.group(1)))", "raise LabelOverflow(token[b], 0)"),
    ("ylong is classed by twice the chord", "ylong=LP.chord_long(y, pos)", "ylong=LP.chord_long(y, 2 * pos)"),
    ("ylong marks every positive pair", "ylong=LP.chord_long(y, pos)", "ylong=(y > 0)"),
    ("gt_out is lost", "gt_out=int(lab[\"gt_out\"])", "gt_out=0"),
    ("the row of a frame is that of the first", "row=int(row[b])", "row=int(row[0])"),
    ("the token of a frame is that of the first", "token=token[b], row=", "token=token[0], row="),
    ("tokens_np s_p is the existence", "s_p=tn[\"s_p\"][b].astype(np.float64)", "s_p=tn[\"e_p\"][b].astype(np.float64)"),
    ("tokens_np e_p is the section start", "e_p=tn[\"e_p\"][b].astype(np.float64)", "e_p=tn[\"s_p\"][b].astype(np.float64)"),
    ("tokens_np flags_p is the existence", "flags_p=tn[\"flags_p\"][b].astype(np.float64)", "flags_p=tn[\"e_p\"][b][:, None].repeat(8, 1).astype(np.float64)"),
    ("tokens_np dirv is the position", "dirv=tn[\"dirv\"][b].astype(np.float64)", "dirv=tn[\"pos\"][b].astype(np.float64)"),
    ("the label gd is lost", "label=dict(gt_id=lab[\"gt_id\"], gd=lab[\"gd\"],", "label=dict(gt_id=lab[\"gt_id\"], gd=np.zeros_like(lab[\"gd\"]),"),
    ("the label flags are those of the aux", "gt_flags=lab[\"gt_flags\"], gt_aux=lab[\"gt_aux\"]", "gt_flags=np.pad(lab[\"gt_aux\"], ((0, 0), (0, 5))), gt_aux=lab[\"gt_aux\"]"),
    ("the token drop is ignored", "keep = keep_nodes(token[b], K, token_drop, seed)", "keep = np.ones(K, bool)"),
    ("the token drop ignores the seed", "keep = keep_nodes(token[b], K, token_drop, seed)", "keep = keep_nodes(token[b], K, token_drop, 0)"),
    ("a removed token stays valid", "valid = tn[\"valid\"][b] & keep", "valid = tn[\"valid\"][b]"),
    ("a removed token keeps its type", "ytype=np.where(keep, lab[\"gt_type\"], 0).astype(np.int64)", "ytype=lab[\"gt_type\"].astype(np.int64)"),
    ("a removed token keeps its pairs", "K, keep if token_drop > 0 else None)", "K, None)"),
    ("tokens_np keeps the removed tokens valid", "cell=tn[\"cell\"][b].astype(np.int64), valid=valid.copy()),", "cell=tn[\"cell\"][b].astype(np.int64), valid=tn[\"valid\"][b].copy()),"),
    ("the source does not pass the token drop on", LAB_CALL, "live_batch(tk, tables, token, row, self.K, 0.0, 0, gate)"),
    ("the source does not pass the gate on", LAB_CALL, "live_batch(tk, tables, token, row, self.K, token_drop, seed)"),
    ("forward drops the token drop and the gate", FWD, "return dict(no=no, batch=self.labelled(tk, tables, token, row))"),
    ("the node targets are transposed", "return {k: g[k] for k in NODE_KEYS}", "return {k: np.swapaxes(g[k], -1, -2) for k in NODE_KEYS}"),
    ("B's loss runs on tokens without the graph", "return QG.node_loss(no, targets)", "return QG.node_loss({k: v.detach() for k, v in no.items()}, targets)", "does not require grad"),
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
    ("compare_frame ignores s_p of tokens_np", TNP, TNP.replace("for k in NP_KEYS})", "for k in NP_KEYS if k != \"s_p\"})")),
    ("compare_frame ignores the valid of tokens_np", "np_valid=int((tok(a[\"tnp\"], \"valid\", ja) != tok(w[\"tnp\"], \"valid\", jw)).sum())", "np_valid=0"),
    ("the fixture compares tokens_np with itself", "tnp=o[\"tokens_np\"]", "tnp=live[\"tnp\"]"),
    ("the fixture ignores the label counts", "stats += int(out[-1][\"same_order\"] and any(", "stats += int(False and any("),
    ("the fixture accepts nothing to compare", "assert n > 0 and len(seeds) > 0 and bs > 0, f\"nothing to compare", "assert True, f\"nothing to compare"),
    ("the GPU fixture accepts a short last batch", "assert n % bs == 0 or not gpu, f\"--gpu: n", "assert True, f\"--gpu: n"),
]


def run_selftest():
    r = subprocess.run([sys.executable, "-c", LAUNCH, os.path.join(DATA, MOD), "selftest"], capture_output=True, text=True, env=ENV)
    assert r.returncode == 0, f"{MOD} selftest failed:\n{r.stdout[-800:]}{r.stderr[-2500:]}"
    print(r.stdout.strip().splitlines()[-1], flush=True)


def run_real():
    """the real-data commands: the node targets against QData and the cache's checkpoint (VAL and TRAIN, 20 frames each), the fixture on 8 frames of both seeds at bs 4 and of one seed at bs 8 (exit code 0 = the label step on the cached tokens is exactly the cache at any
    batch size) and the gradient check of the card on 2 real frames with the champion head (the head's loss reaches every node-head parameter of the real B', none of the frozen stages, the pyramid is unchanged by a step)"""
    for args in (["check", "--split", "val", "--n", "20"], ["check", "--split", "train", "--n", "20"], ["fixture", "--n", "8", "--seeds", "0", "1"], ["fixture", "--n", "8", "--seeds", "1", "--bs", "8"], ["grad", "--n", "2", "--head", "e365_s1_s0"]):
        r = subprocess.run([sys.executable, "-c", LAUNCH, os.path.join(DATA, MOD)] + args, capture_output=True, text=True, env=ENV)
        assert r.returncode == 0, f"{MOD} {' '.join(args)} failed:\n{r.stdout[-1500:]}{r.stderr[-1500:]}"
        print(f"{' '.join(args)}: {r.stdout.strip().splitlines()[-1]}", flush=True)


def one(args):
    k, (name, a, b, *expect), tmp = args
    with open(os.path.join(DATA, MOD), encoding="utf-8") as fh:
        src = fh.read()
    if src.count(a) != 1:
        return name, f"NOT APPLIED ({src.count(a)} matches)"
    d = os.path.join(tmp, str(k))
    os.makedirs(d)
    p = os.path.join(d, MOD)
    with open(p, "w", encoding="utf-8") as fh:
        fh.write(src.replace(a, b))
    r = subprocess.run([sys.executable, "-c", LAUNCH, p, "selftest"], capture_output=True, text=True, env=ENV)
    if r.returncode == 0:
        return name, "SURVIVED"
    if (re.compile(expect[0]) if expect else ASSERTED).search(r.stderr + r.stdout):
        return name, "killed"
    return name, "CRASHED: " + (r.stderr.strip().splitlines() or ["no output"])[-1][:140]


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
        assert not left, f"mutants survived or crashed: {left}"
    print("test_lane_live ok")
