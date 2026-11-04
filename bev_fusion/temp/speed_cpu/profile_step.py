"""one keyframe forward + loss + backward of the t20 chain under torch.profiler (CPU), ops per section, forward and backward.
usage: profile_step.py B n_boxes tag [small]     (small = 24 m x 24 m grid: less memory, the op counts per section are the same, only B scaling is read from it)"""
import collections
import json
import re
import sys
import time

import common as C
import torch
from torch.profiler import ProfilerActivity, profile, record_function

B, NB, TAG = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
SMALL = len(sys.argv) > 4 and sys.argv[4] == "small"
grid_kwargs = {"x_max": 24.0, "y_min": -12.0, "y_max": 12.0} if SMALL else None
cfg, model, loss, batch, _ = C.make(B=B, n_boxes=NB, grid_kwargs=grid_kwargs)

import extract  # noqa: E402
import losses  # noqa: E402
import model as M  # noqa: E402
from trainkit.util import detach_tree  # noqa: E402

# ---- sections: module hooks (children of the model, the model, the loss) and wrapped functions
calls = collections.Counter()
live = []


def hook_pair(name, mod):
    def pre(m, a, k=None):
        calls[name] += 1
        rf = record_function(f"SEC::{name}#{calls[name]}" if name in ("sem_head", "obj_head", "head_1m") else f"SEC::{name}")
        rf.__enter__()
        live.append(rf)

    def post(m, a, o, k=None):
        live.pop().__exit__(None, None, None)
    mod.register_forward_pre_hook(pre)
    mod.register_forward_hook(post)


for n, m in model.named_children():
    hook_pair(n, m)
hook_pair("model.forward", model)
hook_pair("loss.forward", loss)


def wrap(mod, name, label):
    f = getattr(mod, name)

    def w(*a, **k):
        with record_function("SEC::" + label):
            return f(*a, **k)
    setattr(mod, name, w)


wrap(losses, "build_layer_targets", "loss.build_layer_targets")
wrap(losses, "map_terms", "loss.map_terms")
wrap(losses, "object_head_terms", "loss.object_head_terms")
wrap(losses, "motion_term", "loss.motion_term")
wrap(M, "dense_boxes", "model.dense_boxes")
import splat as SPL
wrap(SPL, "splat", "splat.splat")
wrap(SPL, "chain_positions", "splat.chain_positions")
wrap(losses, "build_targets", "objhead.build_targets(host)")
wrap(losses, "head_loss", "objhead.head_loss")
wrap(M, "soft_line_map", "model.soft_line_map")
wrap(M.Precompute, "__call__", "model.precompute")


def one(label, state):
    model.zero_grad(set_to_none=True)
    for q in loss.parameters():
        q.grad = None
    prof = profile(activities=[ProfilerActivity.CPU], record_shapes=False, with_stack=True)
    t0 = time.time()
    with prof:
        with record_function("PHASE::forward"):
            with record_function("SEC::forward_loss.glue"):
                out, res, _ = C.T.forward_loss(model, loss, batch, 0.5, "cpu", "off", cfg, state=state, step=(1 / 12, 8.0, 0.05) if state is not None else None)
        with record_function("PHASE::backward"):
            out["total"].backward()
    t1 = time.time() - t0
    return prof, res, t1


def analyse(prof, label):
    evs = list(prof.events())
    by_id = {id(e): e for e in evs}
    META = {"aten::view", "aten::reshape", "aten::_reshape_alias", "aten::expand", "aten::permute", "aten::transpose", "aten::t", "aten::select", "aten::slice",
            "aten::squeeze", "aten::unsqueeze", "aten::as_strided", "aten::alias", "aten::detach", "aten::detach_", "aten::empty", "aten::empty_like", "aten::empty_strided",
            "aten::new_empty", "aten::new_empty_strided", "aten::resolve_conj", "aten::resolve_neg", "aten::_unsafe_view", "aten::view_as", "aten::unflatten", "aten::flatten",
            "aten::split", "aten::unbind", "aten::narrow", "aten::movedim", "aten::swapaxes", "aten::to", "aten::numel", "aten::size", "aten::stride", "aten::is_floating_point",
            "aten::chunk", "aten::split_with_sizes", "aten::expand_as", "aten::view_as_real", "aten::type_as", "aten::contiguous", "aten::lift_fresh", "aten::result_type",
            "aten::is_same_size", "aten::dim", "aten::mT", "aten::_has_compatible_shallow_copy_type", "aten::set_", "aten::real", "aten::imag", "aten::ravel"}
    SYNC = {"aten::item", "aten::_local_scalar_dense", "aten::nonzero", "aten::masked_select", "aten::equal", "aten::is_nonzero", "aten::allclose", "aten::unique_consecutive",
            "aten::_unique2", "aten::nonzero_numpy", "aten::argwhere", "aten::bincount", "aten::masked_scatter", "aten::repeat_interleave"}

    def children(e):
        return e.cpu_children

    def leaf(e):
        return e.name.startswith("aten::") and not any(c.name.startswith("aten::") or children_has_aten(c) for c in children(e))

    memo = {}

    def children_has_aten(e):
        k = id(e)
        if k not in memo:
            memo[k] = any(c.name.startswith("aten::") or children_has_aten(c) for c in children(e))
        return memo[k]

    def section(e):
        p = e
        while p is not None:
            if p.name.startswith("SEC::"):
                return p.name[5:]
            p = p.cpu_parent
        return "(none)"

    def phase(e):
        p = e
        while p is not None:
            if p.name.startswith("PHASE::"):
                return p.name[7:]
            p = p.cpu_parent
        return "(none)"

    def walk(e):
        yield e
        for c in e.cpu_children:
            yield from walk(c)

    # forward: section by ancestor SEC; map sequence numbers
    fwd_sec = {}
    for e in evs:
        if phase(e) == "forward" and e.sequence_nr >= 0:
            fwd_sec[e.sequence_nr] = section(e)
    rows = collections.defaultdict(lambda: collections.Counter())
    sync_rows = collections.defaultdict(collections.Counter)
    sync_sites = collections.defaultdict(collections.Counter)
    n_unmatched = 0
    for e in evs:
        if not leaf(e) or e.name in META:
            continue
        ph = phase(e)
        if ph == "forward":
            sec = re.sub(r"#\d+$", "", section(e)) if False else section(e)
            rows[sec]["fwd"] += 1
        elif ph == "backward":
            p = e
            sq = None
            while p is not None:                     # the autograd node that runs this op
                if p.name.startswith("autograd::engine::evaluate_function") and p.sequence_nr >= 0:
                    sq = p.sequence_nr
                    break
                p = p.cpu_parent
            sec = fwd_sec.get(sq) if sq is not None else None
            if sec is None:
                sec = "(bwd unmatched / accumulate)"
                n_unmatched += 1
            rows[sec]["bwd"] += 1
        if e.name in SYNC and ph == "forward":
            sync_rows[section(e)][e.name] += 1
    # sync sites with python stack (forward ops)
    for e in evs:
        if e.name in SYNC and phase(e) == "forward":
            st = [s for s in (e.stack or []) if "/training/" in s or "/trainkit/" in s or "test_slots" in s]
            site = st[0].split("/")[-1] if st else "(no repo frame)"
            sync_sites[(section(e), e.name)][site] += 1
    tot = {"fwd": sum(r["fwd"] for r in rows.values()), "bwd": sum(r["bwd"] for r in rows.values())}
    ncalls = collections.Counter(re.sub(r"#\d+$", "", e.name[5:]) for e in evs if e.name.startswith("SEC::"))
    print("  calls per keyframe:", dict(ncalls))
    return rows, tot, sync_rows, sync_sites


results = {}
state0 = None
for label in ("cold", "warm"):
    prof, res, t = one(label, state0)
    rows, tot, sync_rows, sync_sites = analyse(prof, label)
    results[label] = {"rows": {k: dict(v) for k, v in rows.items()}, "tot": tot, "sync": {k: dict(v) for k, v in sync_rows.items()},
                      "sites": {f"{k[0]} | {k[1]}": dict(v) for k, v in sync_sites.items()}, "time_s": t}
    print(f"\n=== {label}: B={B} boxes/sample={NB} grid={'small' if SMALL else 'full'}  kernel-like leaf aten ops: forward {tot['fwd']}, backward {tot['bwd']}  ({t:.1f} s with the profiler)")
    for sec, r in sorted(rows.items(), key=lambda kv: -(kv[1]["fwd"] + kv[1]["bwd"])):
        print(f"  {sec:34s} fwd {r['fwd']:6d}  bwd {r['bwd']:6d}  share {100 * (r['fwd'] + r['bwd']) / (tot['fwd'] + tot['bwd']):5.1f} %")
    print("  sync-like ops in the forward (name per section):")
    for sec, c in sorted(sync_rows.items(), key=lambda kv: -sum(kv[1].values())):
        print(f"    {sec:34s} {dict(c)}")
    state0 = detach_tree(res["state"])
    del prof
json.dump(results, open(f"temp/speed_cpu/profile_{TAG}.json", "w"), indent=1)
