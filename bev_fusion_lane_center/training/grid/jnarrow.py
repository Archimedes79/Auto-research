"""E353 (D99 step 1): the junction maps head narrowed PHYSICALLY by channel surgery on a trained `jends_train.JEndModel`, the step after the sensitivity table of E345 (`evaluation/jprune_probe.py`, which only replaces channels by constants).

    import jnarrow as JN
    spec = {"calib": "temp/e345/e324_s_s0/calib.pt", "edits": [{"unit": "jm.m05.hid", "frac": 0.75}]}       # plain JSON; frac = the REMOVED share
    model, args = JN.build_narrowed("e324_s_s0", spec)     # the stored model, groups removed, means folded; model.narrow_spec = the resolved spec (a checkpoint stores it as args["narrow"])
    model, args = JN.load_model("e353_n1_s0", "cpu")       # `jends_eval.load_model`'s contract for a narrowed (or an unnarrowed) checkpoint: the shapes come from args["narrow"]
    wsl bash tools/env/_py14.sh test_jnarrow.py            # the tests (CPU, real checkpoint and VAL frames)

Rule. An edit names a unit of the probe (kind `chan`) or a tensor (`tap`) of the maps head; `frac` of its GroupNorm groups (`int(frac * G + 0.5)` of the G groups) are removed, the groups with the LOWEST summed E345 importance s_c (`calib.pt`, stable ties), whole groups
only: the kept channels keep the statistics of their group, so the kept function does not change (an arbitrary 25 % of the channels would regroup GroupNorm(8, 32) into GroupNorm(8, 8) and change every kept channel). The channel set therefore differs from the probe's per-channel
ranking; the cost of the removed set is `jprune_cost.Cost`'s (same channels). Surgery per tap: the producer conv loses the removed output channels, its GroupNorm(G, C) becomes GroupNorm(G - removed groups, C - removed) with the kept affine parameters, every consumer conv loses
the removed input columns and gets the removed channels' calibration means (post-ReLU) folded into a bias (`sum_c mean_c * sum_kernel W[:, c]`; a consumer without bias gets a NEW bias parameter: one more tensor in the state dict). Exact in the interior of the map; at the zero-padding
border the hook-ablated model sees zeros outside the map where the folded bias sees the constant (a one-pixel ring of the consumer's output; the next GroupNorm sees it in its statistics). Only maps-head tensors with a GroupNorm producer and conv consumers are built; the
residual stream (`jm.p2`, `jm.blocks.i`), head outputs, the pair head (linears: no MACs worth narrowing) and the pyramid (needs a pyramid forward with gradient and the node head's input columns: E352) are refused. The state of a narrowed model is a plain state dict;
`build_narrowed` / `load_model` rebuild the shapes from the spec (`resolve` stores the removed channels and the group counts, so no calibration is needed to read a checkpoint).
"""
import copy
import hashlib
import os
import sys

import torch
import torch.nn as nn

ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for sub_ in ("evaluation", "training", "training/grid", "training/data", "training/data/prep", "training/train"):
    sys.path.insert(0, os.path.normpath(os.path.join(ROOT, sub_)))
import jends_eval as JEV  # noqa: E402
import jprune_cost as PC  # noqa: E402
import jprune_probe as PP  # noqa: E402
from jprune_cost import Edit  # noqa: E402

_STORED_LOAD = JEV.load_model                                                           # the loader of the stored runs, kept for the checkpoints without a spec (the equality of the unnarrowed dump rests on it)
assert _STORED_LOAD.__module__ == "jends_eval", "jnarrow must be imported before the loader of jends_eval is replaced"


# ---------------------------------------------------------------- 1. wiring and the checks of what is built
def producer_of(W, tap):
    """the layer that writes the tensor `tap` (None when there is not exactly one)"""
    ps = [layer for layer, (_, out, _) in W.items() if out == tap]
    return ps[0] if len(ps) == 1 else None


def width(model, W, tensor):
    return model.jmodel.pyr.c_in if tensor == "x" else PC.sub(model, producer_of(W, tensor)).out_channels


def readers(model, W, tap):
    """[(layer, [column offsets of the tap's occurrences in its input])] of every layer that reads the tap (the inputs are concatenated in the order of the wiring). The widths are those of `model` AS IT IS: `narrow` cuts the taps one after the other, so a layer that
    reads two edited tensors has already lost the first one's columns when the second is cut, and the offsets must match its current input (offsets from the unnarrowed model would slice the wrong columns; test_two_edited_taps_of_one_reader)"""
    return [(layer, [sum(width(model, W, t) for t in ins[:k]) for k, t in enumerate(ins) if t == tap]) for layer, (ins, _, _) in W.items() if tap in ins]


def put(model, path, new):
    parent, _, last = path.rpartition(".")
    host = PC.sub(model, parent)
    if last.isdigit():
        host[int(last)] = new
    else:
        setattr(host, last, new)


def group_norm_of(model, W, tap):
    """the GroupNorm right behind the producer conv of the tap (the block is conv -> GroupNorm -> ReLU)"""
    path = producer_of(W, tap)
    parent, _, i = path.rpartition(".")
    seq = PC.sub(model, parent)
    return seq[int(i) + 1] if isinstance(seq[int(i) + 1], nn.GroupNorm) else None


def check_tap(model, W, tap):
    """refuses what is not built (see the docstring); -> (producer path, GroupNorm, readers)"""
    if not tap.startswith("jm."):
        raise NotImplementedError(f"{tap}: only tensors of the maps head (jm.*) are built; the pyramid is E352 (pyramid forward with gradient, the node head's input columns), the pair head carries no MACs")
    if tap in {"jm.p2"} | {f"jm.blocks.{i}" for i in range(len(model.jmodel.jm.blocks))}:
        raise NotImplementedError(f"{tap} is read by a residual add: the stream needs one channel set in every tensor of it")
    path = producer_of(W, tap)
    gn = group_norm_of(model, W, tap) if path else None
    rd = readers(model, W, tap)
    ok = path and isinstance(PC.sub(model, path), nn.Conv2d) and gn is not None and rd and all(isinstance(PC.sub(model, layer), nn.Conv2d) and PC.sub(model, layer).groups == 1 for layer, _ in rd)
    if not ok:
        raise NotImplementedError(f"{tap}: not a hidden tensor behind a conv + GroupNorm with conv readers (head outputs and linears are not built)")
    return path, gn, rd


# ---------------------------------------------------------------- 2. the spec: which groups go
def sha256_file(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def resolve(model, spec):
    """-> the spec with `taps` filled: {tap: {removed: [channels], groups: [kept groups, group size]}} chosen from the calibration file `spec["calib"]` (its sha256 is stored in `calib_sha256`); a resolved spec is returned as it is, an empty one
    (no edits) has no taps. spec = {"calib": path, "edits": [{"unit" | "tap": name, "frac": removed share}]}"""
    if "taps" in spec:
        return spec
    if not spec.get("edits"):
        return dict(spec, taps={})
    path = PP.rootpath(spec["calib"])
    calib = PP.load_calib(path, "cpu")
    units, W, out = {u.name: u for u in PP.build_units(model)}, PC.wiring(model), {}
    for e in spec["edits"]:
        assert ("unit" in e) != ("tap" in e) and set(e) <= {"unit", "tap", "frac"}, f"an edit names a unit or a tap and a frac: {e}"
        f = float(e["frac"])
        assert 0.0 < f < 1.0, e
        if "unit" in e:
            u = units[e["unit"]]
            if u.kind != "chan" or u.idx:
                raise NotImplementedError(f"{u.name}: only channel units are built")
            names = u.taps
        else:
            names = (e["tap"],)
        for tap in names:
            assert tap not in out, f"{tap} is edited twice"
            _, gn, _ = check_tap(model, W, tap)
            G, C = gn.num_groups, gn.num_channels
            gs, n_rm = C // G, int(f * G + 0.5) - 1
            assert 0 < n_rm < G, f"{tap}: {f} of {G} groups is {n_rm}: nothing or everything would go"
            score = calib.s[tap].double().view(G, gs).sum(1)                           # summed importance of the channels of a group
            gone = sorted(torch.argsort(score, stable=True)[:n_rm].tolist())
            out[tap] = dict(removed=[c for g in gone for c in range(g * gs, (g + 1) * gs)], groups=[G - n_rm, gs])
    return dict(spec, taps=out, calib_sha256=sha256_file(path))


# ---------------------------------------------------------------- 3. the surgery
def _conv_like(old, cin, cout, bias):
    return nn.Conv2d(cin, cout, old.kernel_size, old.stride, old.padding, old.dilation, bias=bias, padding_mode=old.padding_mode).to(old.weight.device, old.weight.dtype)


def narrow_tap(model, W, tap, removed, mean):
    """removes the channels `removed` (whole GroupNorm groups) of the tensor `tap`; mean (C,) = their constant to fold into the readers' biases (None: zeros, for a checkpoint whose weights are loaded afterwards)"""
    path, gn, rd = check_tap(model, W, tap)
    conv = PC.sub(model, path)
    C, gs = conv.out_channels, gn.num_channels // gn.num_groups
    gone = set(removed)
    assert len(gone) == len(removed) and all(set(range(g * gs, (g + 1) * gs)) <= gone for g in {c // gs for c in gone}), f"{tap}: whole GroupNorm groups only"
    keep = [c for c in range(C) if c not in gone]
    mean = torch.zeros(C) if mean is None else mean.detach().cpu()
    new = _conv_like(conv, conv.in_channels, len(keep), False)
    new.weight.data.copy_(conv.weight.data[keep])
    ng = nn.GroupNorm(len(keep) // gs, len(keep), eps=gn.eps).to(conv.weight.device, conv.weight.dtype)
    ng.weight.data.copy_(gn.weight.data[keep])
    ng.bias.data.copy_(gn.bias.data[keep])
    put(model, path, new)
    put(model, path.rpartition(".")[0] + f".{int(path.rpartition('.')[2]) + 1}", ng)
    for layer, offs in rd:
        r = PC.sub(model, layer)
        cols = sorted(o + c for o in offs for c in removed)
        kept = [j for j in range(r.in_channels) if j not in set(cols)]
        fold = sum((r.weight.data[:, [o + c for c in removed]].double() * mean[removed].double().view(1, -1, 1, 1)).sum((1, 2, 3)) for o in offs)
        nr = _conv_like(r, len(kept), r.out_channels, True)                                       # a reader without bias gets a new bias parameter
        nr.weight.data.copy_(r.weight.data[:, kept])
        nr.bias.data.copy_(((r.bias.data.double() if r.bias is not None else 0.0) + fold).to(r.weight.dtype))
        put(model, layer, nr)



def narrow(model, spec, fold=True):
    """the surgery of the spec on `model`, IN PLACE; -> model with `model.narrow_spec` = the resolved spec. fold=False builds the shapes only (the weights of a checkpoint are loaded afterwards; no calibration file is read for a resolved spec)"""
    spec = resolve(model, spec)
    W = PC.wiring(model)
    mean = PP.load_calib(PP.rootpath(spec["calib"]), "cpu").mean if fold and spec["taps"] else {}
    if fold and spec["taps"] and "calib_sha256" in spec:
        assert sha256_file(PP.rootpath(spec["calib"])) == spec["calib_sha256"], "the calibration file is not the one the spec was resolved on"
    for tap, t in spec["taps"].items():
        narrow_tap(model, W, tap, t["removed"], mean.get(tap))
        gn = group_norm_of(model, W, tap)
        assert [gn.num_groups, gn.num_channels // gn.num_groups] == t["groups"], (tap, gn, t)
    model.narrow_spec = spec
    return model


# ---------------------------------------------------------------- 4. building and loading
def model_of_args(a):
    """the full `JEndModel` of a checkpoint's args, as `jends_eval.load_model` builds it (the test compares the two)"""
    import jends_train as JT
    return JT.JEndModel(a["maps_args"], order=not a["no_order"], evidence=not a["no_evidence"], scales=bool(a.get("scales", False)), ntok=bool(a.get("ntok", False)), unc=bool(a.get("unc", False)), attach=bool(a.get("attach", False)))


def load_model(run, dev="cpu"):
    """`jends_eval.load_model`'s contract -> (model on dev in eval mode, args) for a stored run, narrowed or not (the shapes come from args["narrow"]); the root is `jends_eval.ROOT` (the tests move it)"""
    ck = torch.load(os.path.join(JEV.ROOT, "runs", run, "model.pt"), map_location="cpu")
    a = ck["args"]
    if not a.get("narrow", {}).get("taps"):
        return _STORED_LOAD(run, dev)
    m = narrow(model_of_args(a), a["narrow"], fold=False)
    m.load_state_dict(ck["state"])
    return m.to(dev).eval(), a


def build_narrowed(base_run, spec, dev="cpu"):
    """-> (the stored model of `base_run` with the surgery of the spec, its args): the starting point of the fine-tune and the n1z row"""
    m, a = _STORED_LOAD(base_run, dev)
    return narrow(m, spec).eval(), a


# ---------------------------------------------------------------- 5. what the surgery costs and what it changes
def edits_of(spec, calib=None):
    """the probe's edits (`jprune_probe.Edit`) of a resolved spec: the same channels; with `calib` the replacement values, so that `jprune_probe.ablate` removes them by hook"""
    return [Edit("chan", tap=t, idx=tuple(v["removed"]), val=None if calib is None else calib.mean[t][v["removed"]]) for t, v in spec["taps"].items()]


def expected_saving(full_model, spec):
    """what `jprune_cost.Cost` says the removal of the spec saves (dict macs, params, macs_pyr, macs_maps, macs_pair, macs_mha), counted on the full model"""
    return PC.Cost(copy.deepcopy(full_model).eval()).saved(edits_of(spec))


def measured_macs(model):
    """GMACs per frame by torch FlopCounterMode (`jmodel_cost.cost`: conv / linear / matmul, attention modules not counted): dict(pyr, maps, pair, total); `total` includes B's node head"""
    import jmodel_cost as JC
    c = JC.cost(model)
    return dict(pyr=c["pyramid (B's, frozen)"]["macs_g"], maps=c["junction maps head"]["macs_g"], pair=c["pair head"]["macs_g"], total=c["total"]["macs_g"])


@torch.no_grad()
def difference(full_model, narrowed, batches, kp_tol=0.05):
    """the narrowed model against the hook-ablated model (E345's `ablate`) that replaces the SAME channels by their calibration means, on batches of x: max / mean absolute difference of the entry / exit heat logits (`j1[:, 2:4]`), the paint maps `j05`,
    the keypoint positions; the fraction of changed keypoints (valid slots whose position moved by more than `kp_tol` metres or whose validity flipped); and, per consumer conv of the removed tensors, the largest difference of its OUTPUT inside the map and on its
    one-pixel border ring. -> dict"""
    spec = narrowed.narrow_spec
    calib = PP.load_calib(PP.rootpath(spec["calib"]), "cpu")
    ablated, _ = PP.ablate(full_model, edits_of(spec, calib), PP.tap_table(full_model))
    W = PC.wiring(full_model)
    seen, hooks = {}, []
    for name, m in (("ablated", ablated), ("narrowed", narrowed)):
        for tap in spec["taps"]:
            for layer, _ in readers(full_model, W, tap):
                hooks.append(PC.sub(m, layer).register_forward_hook(lambda _m, _a, o, k=(name, layer): seen.__setitem__(k, o)))
    acc = dict(heat_max=0.0, heat_sum=0.0, heat_n=0, paint_max=0.0, paint_sum=0.0, paint_n=0, paint_inner_max=0.0, kp_max=0.0, kp_changed=0, kp_n=0, logit_max=0.0, logit_sum=0.0, logit_flips=0, logit_n=0, reader_interior=dict(), reader_border=dict())
    for x in batches:
        _, oa, ea = ablated(x, gt=None)
        _, on, en = narrowed(x, gt=None)
        for key, (u, v) in dict(heat=(oa["j1"][:, 2:4], on["j1"][:, 2:4]), paint=(oa["j05"], on["j05"])).items():
            d = (u - v).abs()
            acc[key + "_max"] = max(acc[key + "_max"], float(d.max()))
            acc[key + "_sum"] += float(d.sum())
            acc[key + "_n"] += d.numel()
        acc["paint_inner_max"] = max(acc["paint_inner_max"], float((oa["j05"] - on["j05"])[..., 2:-2, 2:-2].abs().max()))
        ok = ea["allowed"] & en["allowed"]
        dl = (ea["logit"].float() - en["logit"].float())[ok].abs()
        acc["logit_max"] = max(acc["logit_max"], float(dl.max()) if dl.numel() else 0.0)
        acc["logit_sum"] += float(dl.sum())
        acc["logit_flips"] += int(((ea["logit"] > 0) != (en["logit"] > 0))[ok].sum())
        acc["logit_n"] += int(ok.sum())
        va, vn = ea["valid"].flatten(1), en["valid"].flatten(1)
        dp = (ea["kp_pos"] - en["kp_pos"]).norm(dim=-1)
        both = va & vn
        acc["kp_max"] = max(acc["kp_max"], float(dp[both].max()) if both.any() else 0.0)
        acc["kp_changed"] += int((va != vn).sum() + (both & (dp > kp_tol)).sum())
        acc["kp_n"] += int((va | vn).sum())
        for (name, layer), o in list(seen.items()):
            if name == "narrowed":
                d = (seen[("ablated", layer)] - o).abs()
                inner = d[..., 1:-1, 1:-1].max()
                ring = torch.cat([d[..., 0, :].flatten(), d[..., -1, :].flatten(), d[..., :, 0].flatten(), d[..., :, -1].flatten()]).max()
                acc["reader_interior"][layer] = max(acc["reader_interior"].get(layer, 0.0), float(inner))
                acc["reader_border"][layer] = max(acc["reader_border"].get(layer, 0.0), float(ring))
    for h in hooks:
        h.remove()
    return dict(heat_max=acc["heat_max"], heat_mean=acc["heat_sum"] / acc["heat_n"], paint_max=acc["paint_max"], paint_mean=acc["paint_sum"] / acc["paint_n"], paint_inner_max=acc["paint_inner_max"], logit_max=acc["logit_max"], logit_mean=acc["logit_sum"] / max(acc["logit_n"], 1),
                logit_sign_flips=acc["logit_flips"] / max(acc["logit_n"], 1), kp_pos_max_m=acc["kp_max"], kp_changed_frac=acc["kp_changed"] / max(acc["kp_n"], 1),
                kp_slots=acc["kp_n"], reader_interior_max=acc["reader_interior"], reader_border_max=acc["reader_border"])
