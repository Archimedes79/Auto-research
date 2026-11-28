"""The lane network as ONE module: the frozen GaussianCaR features of a frame -> the lane head B' (`qgraph.QGraph`: pyramid, node head, Matrix-NMS
layer; its link, section and assignment heads ride in the checkpoint for stage 0) -> the K cells with the largest existence as tokens -> the token
head (`jtoken.JTokenHead`) -> the delivered graph (`lane_readout`). The trainers, the predict step, the token cache and the evaluation build, load and
run the network here and nowhere else.

    net, ck = LaneNet.from_checkpoint("<run>/model.pt")        # stage 2 (state, b_state, b_args, args); stage 1 (state, args) with b_run=<its B' run>
    net = LaneNet(load_b("<runs>/e292_a_s0"), build_head(args), K=512, train_nodes=True)     # from the parts
    b = build_b(args)                                          # a new B' of a stage-0 run's args (training/train/qgraph_train.py)
    no, tk = net.tokens(x)                                     # x (B, 39, 200, 200) -> the node-stage outputs and the K tokens, graph kept
    out, tk = net(x)                                           # the head's output on the live tokens
    preds = net.graph(x, decode)                               # the delivered graph of every frame (qformat predictions), no_grad
    batch = net.labelled(tk, tables, token, row, gate)         # the head's training batch of live tokens (`lane_live_data.live_batch`)
    net.head.parameters(), net.node_params(), net.pyr_params()  # the three groups of the optimiser and of the clip

Freezing: B' is frozen except its node head (`train_nodes`) and, with `train_pyr`, its pyramid; B' stays in eval mode except the node head, which
follows the module; the pyramid runs under no_grad unless `train_pyr`. B' runs in float32, also inside a trainer's bf16 autocast (as the token cache).
Tokens (`live_tokens`): `lane_tokcache.select_tokens` (a stable argsort by the existence logit: an index choice, every gather keeps the graph), `tok`
rounded to float16 as the cache stores it, the gradient passed straight through. On the GPU at batch 4 the tokens equal the cache bit for bit.
Head: the champion's architecture `ARCH`; `build_head` reads live_inputs and the soft output layers of a run's args. Options that no longer exist
(`REMOVED`: the tag that keeps their code -> option -> the value a stored run must hold) load when a run holds that value or nothing, another value is
refused with the option's name and the tag (`check_removed`; `qgraph.from_args` does the same for B').
Checkpoints as the trainers write them: stage 1 `state` (the head) + `args`; stage 2 adds `b_state` (the whole `QGraph` state), `b_args` and `rec`; a
B' run `state` + `args`. Tests: net/tests/test_lane_net.py, net/tests/test_e2e_reach.py, net/tests/test_jtoken.py"""
import contextlib
import json
import os

import torch
import torch.nn as nn

from lane_center.training import bflc_paths as BP
from lane_center.net.model import jtok as JT
from lane_center.net.model import jtoken as JK
from lane_center.net.prep import lane_live_data as LLD
from lane_center.net.model import lane_readout as RO
from lane_center.net.prep import lane_tokcache as LC
from lane_center.net.model import qgraph as QG

ARCH_DEFAULTS = dict(none_weight=1.0, head_seed=None, live_inputs=False)   # what a run of before these options means (the head of the first build)
ARCH = dict(d=64, heads=4, layers=3, ffn=128)                               # the champion's architecture (the head of the first build), a constant since D117
# Options that no longer exist: the tag that keeps their code -> option -> the value a stored run must hold (or nothing): today's head and loss are then that
# run's (`check_removed`)
# D117
REMOVED = {"archive/jtoken-before-simplify-2026-11-17": dict(prior="none", teacher="none", init=None, token_drop=0.0, none_weight_near=None, **ARCH),
           # E376, E379, E381, E383 (K: the one cache's)
           "v1": dict(pos_loss="sum", distill_from=None, distill_weight=1.0, distill_sha256=None, noise_pos=0.0, noise_e=0.0, cache=None, K=512,
                      # E392, E395
                      rounds=0, wire_weight=0.0, msg="real", pos_head=False, curve_w=0.0, node_w=0.0),
           # E399.thin (TIE): a hard token cut
           "archive/e399-thinning-2026-11-28": dict(min_e=None)}


def check_removed(args, where):
    """a run's args that hold an option of REMOVED at another value than the one named there are refused: that head or loss no longer exists (the message names
    the options and the tag that keeps them)"""
    for tag, options in REMOVED.items():
        held = {k: args[k] for k, v in options.items() if args.get(k) not in (None, v)}
        assert not held, f"{where}: {held}: options that no longer exist (the tag {tag} keeps them)"


def to_device(b, dev):
    return {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in b.items()}


def build_head(args):
    """the head of a run's args (live_inputs and the soft output layers skip_suppress (E399), close_gaps (E420), fork_gate (E423), refine_positions (E424):
    absent = the default; the architecture `arch`): the one place that builds it, for the trainer, `load_model` and the stage-2 trainer"""
    d, heads, layers, ffn = arch(args)
    return JK.JTokenHead(d=d, heads=heads, layers=layers, ffn=ffn, skip=args.get("skip_suppress"), gap=args.get("close_gaps"), fork=args.get("fork_gate"), refine=args.get("refine_positions"),
                         live_inputs=bool(args.get("live_inputs", ARCH_DEFAULTS["live_inputs"])))


def arch(args):
    """(d, heads, layers, ffn) = the constants ARCH; the args of a removed option at another value are refused (`check_removed`)"""
    check_removed(args, "the run's args")
    return ARCH["d"], ARCH["heads"], ARCH["layers"], ARCH["ffn"]


def load_checkpoint(path):
    """the dict of a model.pt (or of a trainer's ckpt.pt) on the CPU: with `load_b` the one reader of the network's files"""
    return torch.load(path, map_location="cpu", weights_only=False)


def load_head(run, dev, runs=None):
    """-> (the head of <runs>/<run>/model.pt in eval mode, its args)"""
    ck = load_checkpoint(os.path.join(runs or BP.RUNS, run, "model.pt"))
    check_removed(ck["args"], f"run {run}")
    m = build_head(ck["args"])
    m.load_state_dict(ck["state"])
    return m.to(dev).eval(), ck["args"]


def build_b(args):
    """a new lane head B' of a stage-0 run's args (width, soft_nms: `qgraph.from_args`): what the stage-0 trainer trains (training/train/qgraph_train.py)"""
    return QG.from_args(args)


def load_b(run_dir, device="cpu"):
    """the lane head B' of a run folder: a trained run (`model.pt`: state + args) or, while it trains, its last checkpoint (`ckpt.pt`)"""
    mp = os.path.join(run_dir, "model.pt")
    if os.path.exists(mp):
        sd = torch.load(mp, map_location="cpu")
        a, state = sd.get("args") or {}, sd["state"]
    else:
        sd = torch.load(os.path.join(run_dir, "ckpt.pt"), map_location="cpu", weights_only=False)
        # the trainer writes the run's args here: a running arm with the Matrix-NMS layer must not load as a plain head
        cfg = os.path.join(run_dir, "config.json")
        a, state = (json.load(open(cfg)).get("args") or {}) if os.path.exists(cfg) else {}, sd["model"]
    # the Matrix-NMS layer is a property of the model (args of the checkpoint)
    m = QG.from_args(a)
    m.load_state_dict(state)
    return m.to(device).eval()


class _Round16(torch.autograd.Function):
    """the float16 rounding of `tok` as the cache stores it, the gradient passed straight through (a plain .half().float() would round the gradient to
    float16)"""

    @staticmethod
    def forward(ctx, x):
        return x.half().float()

    @staticmethod
    def backward(ctx, g):
        return g


def live_tokens(no, K, f2_map=None):
    """`lane_tokcache.select_tokens` of the node-stage outputs `no` (the K best cells by e, a stable argsort: an index choice, no gradient through it; every
    gather keeps the graph) with `tok` through `_Round16` -> dict of tensors (B, K, ...); f2_map = the 2 m pyramid map, gathered as `f2` for a cache build that
    stores it"""
    tk = LC.select_tokens(no, f2_map, K)
    tok = no["tok"].flatten(2).gather(2, tk["cell"].long()[:, None, :].expand(-1, no["tok"].shape[1], -1)).transpose(1, 2)
    return dict(tk, tok=_Round16.apply(tok))


def pair_gate(tk, K):
    """the allowed-pair gate of the first K tokens (`jtok.pair_geometry`) as a host array (B, K, K): it feeds statistics of the labels only
    (`lane_live_data.label_batch`)"""
    with torch.no_grad():
        return JT.pair_geometry(tk["pos"][:, :K].detach(), tk["dirv"][:, :K].detach(), tk["valid"][:, :K])[1].cpu().numpy()


class LaneNet(nn.Module):
    """the lane network: `b` = the lane head B' (a `QGraph`), `head` = the token head (None: B' alone, e.g. for a cache build), K = the tokens per frame.
    Everything of B' is frozen except `b.nodes.*` (`train_nodes`) and, with `train_pyr`, `b.pyr.*`; the head's tensors are left as they are (module
    docstring)"""

    def __init__(self, b, head=None, K=512, train_nodes=False, train_pyr=False):
        super().__init__()
        assert train_nodes or not train_pyr, "train_pyr needs train_nodes (the pyramid's gradient reaches it through the node head)"
        self.head, self.b = head, b                                       # the head first: the order of parameters() (a trainer's gradient norm sums in it)
        self.K, self.train_pyr = K, bool(train_pyr)
        for p in self.b.parameters():
            p.requires_grad_(False)
        if train_nodes:
            for p in self.b.nodes.parameters():
                p.requires_grad_(True)
        if self.train_pyr:
            for p in self.b.pyr.parameters():
                p.requires_grad_(True)
        self.train()

    @classmethod
    def from_checkpoint(cls, path, device="cpu", K=512, b_run=None):
        """-> (the network of a stage-1 / stage-2 model.pt, frozen, in eval mode; the checkpoint dict). B' = the run folder `b_run` when given, else the
        architecture of the file's `b_args`; a file that carries a trained B' (`b_state`, a complete `QGraph` state) loads it strictly on top. A stage-1 file
        holds the head only: name its B'"""
        ck = load_checkpoint(path)
        check_removed(ck["args"], path)
        head = build_head(ck["args"])
        head.load_state_dict(ck["state"])
        assert b_run or "b_state" in ck, f"{path} holds no B' (b_state): name the run folder of its B' (b_run)"
        b = load_b(b_run) if b_run else QG.from_args(ck["b_args"])
        if "b_state" in ck:
            b.load_state_dict(ck["b_state"])
        return cls(b, head, K).to(device).eval(), ck

    def train(self, mode=True):
        super().train(mode)
        self.b.eval()
        self.b.nodes.train(mode)
        return self

    def trainable(self):
        return [p for p in self.b.parameters() if p.requires_grad]

    def node_params(self):
        return [p for p in self.b.nodes.parameters() if p.requires_grad]

    def pyr_params(self):
        return [p for p in self.b.pyr.parameters() if p.requires_grad]

    def tokens(self, x, K=None, f2=False, float32=True):
        """pyramid + node stage + `live_tokens` -> (no, tk): the node-stage outputs and the K (default: the network's) live tokens, all with the graph of the
        node head (and of the pyramid with `train_pyr`). Device work only (asynchronous on a GPU: nothing is read back to the host). float32=False: B' follows
        the caller's autocast (only the timing protocol of the record, `jtoken_eval.py timing`)"""
        dev = next(self.b.parameters()).device
        # float32, as the cache was made, also inside the trainer's autocast
        with torch.autocast(device_type=dev.type, enabled=False) if float32 else contextlib.nullcontext():
            with contextlib.nullcontext() if self.train_pyr else torch.no_grad():
                P = self.b.pyr(x.to(dev).float())
            no = self.b.node_stage(P)
            return no, live_tokens(no, self.K if K is None else K, P["feat2"] if f2 else None)

    def labelled(self, tk, tables, token, row, gate=True):
        """the batch of `lane_ptok_data` from the tokens of `tokens` and the GT tables: the label step, which reads the tokens back to the host (a sync) and
        runs on the CPU (`lane_live_data.label_batch`, `LabelOverflow`); gate=False skips the pair gate (its geometry, its device sync and its host copy:
        `n_reach` of the label dicts is then 0, nothing else changes)"""
        with torch.autocast(device_type=tk["pos"].device.type, enabled=False):
            return LLD.live_batch(tk, tables, token, row, self.K, pair_gate(tk, self.K) if gate else None)

    def live(self, x, tables, token, row, gate=True):
        """-> dict(no, batch): `tokens` followed by `labelled`"""
        no, tk = self.tokens(x)
        return dict(no=no, batch=self.labelled(tk, tables, token, row, gate))

    def forward(self, x):
        """-> (the head's output on the live tokens of the features x, the tokens)"""
        no, tk = self.tokens(x)
        return self.head(LLD.head_inputs(tk)), tk

    @torch.no_grad()
    def graph(self, x, decode=RO.D115):
        """features -> the delivered graph of every frame (a qformat prediction) in one forward: `forward`, then the read-out (`lane_readout`: the records of
        the head's output, the decode, the writer); the default decode is the delivered one. NOT the scored path: the evaluation decodes the dump of the predict
        step, which runs B' at batch 4 and the head at batch 8 whatever the batch here (the same nodes and links; the written existence differs by up to 1e-6
        with the head's batch size). Eval mode only: a forward in train mode advances the head's step"""
        assert not self.training, "graph() is the read-out of a network in eval mode"
        out, tk = self(x)
        tn = LLD.host_tokens(tk)
        frames = [LLD.tokens_np(tn, b) for b in range(len(x))]
        recs = RO.batch_dump(out, dict(token=[None] * len(frames), row=[-1] * len(frames), tokens_np=frames))
        return [RO.graph(t, f, t["valid"], decode) for t, f in zip(frames, recs)]
