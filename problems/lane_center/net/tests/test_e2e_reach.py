"""The end-to-end claim of DESIGN R1 as a test: the token loss on the output reaches every module that stage 2 trains, and nothing that is
frozen by design.

    python -P -m lane_center.net.tests.test_e2e_reach          # registered; CPU, 2 threads, about 15 seconds

The network is the champion's stage-2 file (`runs/e372_s4_s0/model.pt` through `LaneNet.from_checkpoint`), the batch two real TRAIN frames
of B' seed 0 (`lane_live_data.LiveData`), the loss the stage-2 trainer's output loss alone (`b_weight` 0): live tokens -> the head ->
the label step -> `loss_jtoken.jtoken_loss` at the champion's none weight. Asserted per parameter tensor, named by its module group:
- `train_nodes 1, train_pyr 1`: every tensor of the token head, the node head and the pyramid gets a finite non-zero gradient, except
  BY_DESIGN (the pyramid's `lines05`: an auxiliary map of stage 0, not on the delivered path); every other tensor of B' (links, map
  encoder, assigner) is frozen and gets none;
- `train_pyr 0`: the same without the pyramid, which gets none;
- the inputs and the hard steps carry no gradient: the GaussianCaR features are a constant input (no parameter of the network outside the
  head and B'), the top-K choice is an index (`cell`, `valid`), the pair gate with the 8 m split is a boolean mask (`cand`), the read-out
  thresholds act on host arrays (`LaneNet.graph`).
Two planted faults must make the check fail: a detach of the live tokens, and a no_grad around the pyramid with `train_pyr 1`."""
import contextlib
import os

import torch

from lane_center.training import bflc_paths as BP
from lane_center.net.loss import loss_jtoken as LJK
from lane_center.net.model import lane_net as LN
from lane_center.net.prep import lane_live_data as LLD

CHAMPION = "runs/e372_s4_s0"                                       # the champion's stage-2 network of seed 0 (champions/lane_center)
FRAMES = (0, 1)                                                    # TRAIN frames in the order of the token cache of B' seed 0
BY_DESIGN = {"b.pyr.lines05.weight", "b.pyr.lines05.bias"}         # trainable with train_pyr, not on the delivered path
TRAINED = {0: ("head.", "b.nodes."), 1: ("head.", "b.nodes.", "b.pyr.")}


def group(name):
    """the module group of a parameter name: head.<layer>, b.<stage>"""
    return ".".join(name.split(".")[:2])


def network(train_pyr):
    """the champion's network with the node head (and with train_pyr the pyramid) trainable, in train mode as stage 2 runs it"""
    net, ck = LN.LaneNet.from_checkpoint(os.path.join(BP.resolve(CHAMPION), "model.pt"))
    net = LN.LaneNet(net.b, net.head, K=net.K, train_nodes=True, train_pyr=bool(train_pyr)).train()
    return net, ck["args"]


def output_loss(net, args, b):
    """the stage-2 step's loss without B''s own node loss (`jlive_train.make_loss` with b_weight 0) -> (loss, tokens, head output)"""
    no, tk = net.tokens(b["x"])
    out = net.head(LLD.head_inputs(tk))
    lb = net.labelled(tk, b["table"], b["token"], b["row"], gate=False)
    assert int(lb["y"].sum()) > 0, "the frames give the head positive pairs"
    loss, _ = LJK.jtoken_loss(out, lb, none_weight=args["none_weight"])
    return loss, tk, out


def gradients(net, args, b):
    """-> ({parameter name: None (no gradient), 'zero', 'reached' or 'non-finite'}, tokens, head output) of one backward of the output loss"""
    for p in net.parameters():
        p.grad = None
    loss, tk, out = output_loss(net, args, b)
    loss.backward()
    state = {}
    for n, p in net.named_parameters():
        if p.grad is None:
            state[n] = None
        elif not bool(torch.isfinite(p.grad).all()):
            state[n] = "non-finite"
        else:
            state[n] = "reached" if bool(p.grad.ne(0).any()) else "zero"
    return state, tk, out


def check_reach(net, state, train_pyr):
    """the rule of R1 on one backward: the trained groups are reached tensor by tensor (BY_DESIGN excepted), every other tensor gets none"""
    trained = {n for n, p in net.named_parameters() if p.requires_grad}
    want = {n for n in state if n.startswith(TRAINED[train_pyr])}
    assert trained == want, f"train_pyr {train_pyr}: the trainable tensors are not those of {TRAINED[train_pyr]}: {sorted(trained ^ want)[:5]}"
    missed = sorted({group(n) for n in want - BY_DESIGN if state[n] != "reached"})
    assert not missed, f"train_pyr {train_pyr}: the output loss does not reach every tensor of {missed}"
    leaked = sorted(n for n, s in state.items() if s not in (None, "zero") and n not in want - BY_DESIGN)
    assert not leaked, f"train_pyr {train_pyr}: tensors frozen by design hold a gradient: {leaked[:5]}"
    frozen = sorted({group(n) for n, s in state.items() if n not in trained and s is not None})
    assert not frozen, f"train_pyr {train_pyr}: frozen groups hold a gradient tensor: {frozen}"


def check_hard_steps(net, b, tk, out):
    """the inputs and the named hard steps carry no gradient"""
    assert not b["x"].requires_grad and b["x"].grad is None, "the GaussianCaR features are a constant input"
    assert {n.split(".")[0] for n, _ in net.named_parameters()} == {"head", "b"}, "the network holds no foundation parameter"
    assert not tk["cell"].dtype.is_floating_point and tk["valid"].dtype == torch.bool, "the top-K choice is an index"
    assert out["cand"].dtype == torch.bool, "the pair gate and the 8 m split are a boolean mask"
    net.eval()
    g = net.graph(b["x"][:1])[0]
    net.train()
    held = [k for part in ("nodes", "links") for k, v in g[part].items() if torch.is_tensor(v)]
    assert not held, f"the read-out (its four thresholds) delivers host arrays, not tensors: {held}"


@contextlib.contextmanager
def planted(fault, net):
    """a fault that cuts a module off: `detach` = the live tokens lose their graph, `pyr_no_grad` = the pyramid runs under no_grad"""
    tokens, pyr_forward = LN.live_tokens, net.b.pyr.forward
    if fault == "detach":
        LN.live_tokens = lambda *a, **k: {key: v.detach() for key, v in tokens(*a, **k).items()}
    else:
        net.b.pyr.forward = torch.no_grad()(pyr_forward)
    try:
        yield
    finally:
        LN.live_tokens = tokens
        if fault != "detach":
            del net.b.pyr.forward


def fails(fn, text):
    """fn must fail with an AssertionError whose message holds `text`"""
    try:
        fn()
    except AssertionError as e:
        assert text in str(e), (text, str(e))
        return
    raise RuntimeError(f"accepted although a module is cut off ({text})")


def main():
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    ds = LLD.LiveData("train", 0, "e292_a")
    b = LLD.collate([ds[i] for i in FRAMES])
    for train_pyr in (1, 0):
        net, args = network(train_pyr)
        state, tk, out = gradients(net, args, b)
        check_reach(net, state, train_pyr)
        reached = sorted({group(n) for n, s in state.items() if s == "reached"})
        frozen = sorted({group(n) for n, s in state.items() if s is None})
        print(f"train_pyr {train_pyr}: reached {reached}; no gradient: {frozen}")
    check_hard_steps(net, b, tk, out)
    net, args = network(1)
    for fault, text in (("detach", "b.nodes"), ("pyr_no_grad", "b.pyr")):
        with planted(fault, net):
            fails(lambda: check_reach(net, gradients(net, args, b)[0], 1), text)
        print(f"planted {fault}: refused ({text} not reached)")
    check_reach(net, gradients(net, args, b)[0], 1)                                    # the faults are gone again
    print("test_e2e_reach ok: the output loss reaches the token head, the node head and the pyramid; the features and the hard steps carry none")


if __name__ == "__main__":
    main()
