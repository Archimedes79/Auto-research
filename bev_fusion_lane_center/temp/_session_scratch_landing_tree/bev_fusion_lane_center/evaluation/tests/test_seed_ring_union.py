"""Spec tests of two loop-22 options of the builder chain (CPU, no GPU):
  * E266 seed ring (`DefaultCfg.lattice_ring`, p16_builder.seed_candidates): the default candidates are the lattice of loops 16 - 21 exactly; the
    ring at 47.5 m adds the 80 points of the square ring after them (inference only) and leaves the seeds of the lattice candidates unchanged;
    the registry entry recall_ring reaches the built network.
  * E264 output ensemble (p16_infer.union_raw, registry kind `union` of evaluate.graphs_of): parents, links and junction heat of joined trees on
    a toy; a seed joined with itself reproduces its graphs on VAL 873 (cached raw outputs, about a minute on the CPU); the registry unions
    join builder entries only.
Run (from the problem folder): wsl bash tools/env/_py14.sh test_seed_ring_union.py"""
import os

import numpy as np
import torch

import p16_builder as PB
import p16_infer as PI
import p16_train as PT


def lattice():
    """the seed lattice of loops 16 - 21 (the code before E266)"""
    g = torch.arange(-45.0, 45.01, 5.0)
    return torch.stack(torch.meshgrid(g, g, indexing="ij"), -1).reshape(-1, 2)


def builder(**kw):
    torch.manual_seed(0)
    return PB.Builder(PB.DefaultCfg(**{**PT.DESIGNS["D7"], **kw}), stem_from=None).eval()


def test_default_candidates_are_the_old_lattice():
    net = builder()
    P, ath = net.seed_candidates(2, "cpu")
    assert ath is None and P.shape == (2, 361, 2) and torch.equal(P[0], lattice()) and torch.equal(P[1], lattice())


def test_ring_adds_the_80_points_of_the_square_ring():
    net = builder(lattice_ring=47.5)
    P = net.seed_candidates(1, "cpu")[0][0]
    assert P.shape == (441, 2) and torch.equal(P[:361], lattice()), "the lattice candidates keep their index"
    side = [-47.5] + [-45.0 + 5.0 * i for i in range(19)] + [47.5]
    want = {(x, y) for x in side for y in side if max(abs(x), abs(y)) == 47.5}  # the 19 lattice positions on each side + 4 corners
    assert len(want) == 80 and sorted(map(tuple, P[361:].tolist())) == sorted(want)
    net.train()
    assert net.seed_candidates(1, "cpu")[0].shape[1] == 361, "training is unchanged (its jitter already reaches the ring)"


def test_ring_leaves_the_lattice_seeds_unchanged():
    torch.manual_seed(1)
    maps, raw_dir = [torch.randn(1, ch, 48, 48) for ch in (32, 32, 16)], torch.randn(1, 2, 48, 48)
    net = builder()
    with torch.no_grad():
        a = net.seeds(maps, raw_dir)
        net.cfg.lattice_ring = 47.5
        b = net.seeds(maps, raw_dir)
    assert b["pos"].shape[1] == 441 and b["pos"][0, 361:].abs().max(-1).values.min() > 44.0  # ring seeds: 47.5 m plus a snap of at most 3.5 m
    for k in ("pos", "th", "val", "h"):
        assert torch.allclose(b[k][:, :361], a[k], atol=1e-6), k


def test_the_ring_entry_reaches_the_built_network():
    """`lattice_ring` of the registry entry recall_ring gets into the network that builder_raw builds and into the assembly config"""
    import evaluate as EV
    e = EV.registry()["recall_ring"]
    net, cfg = PT.build(e["design"], EV.entry_cfg(e))
    assert net.cfg is cfg and cfg.lattice_ring == 47.5 and EV.builder_cfg(e).lattice_ring == 47.5
    assert net.eval().seed_candidates(1, "cpu")[0].shape[1] == 441


def test_registry_unions_join_builders():
    import evaluate as EV
    reg = EV.registry()
    unions = {n: e for n, e in reg.items() if e["kind"] == "union"}
    assert set(unions) >= {"ens_j1rc_23", "ens_j1rc_01", "ens_j1rc_0123", "ens_rc1"}, sorted(unions)
    for n, e in unions.items():
        assert len(e["members"]) >= 2 and all(m in reg and reg[m]["kind"] == "builder" for m in e["members"]), n
        assert len({reg[m]["design"] for m in e["members"]}) == 1 and "tau" in e and "asm" in e, n


def tree(n, parent, links, heat):
    return dict(pos=np.arange(2.0 * n).reshape(n, 2), pi=np.linspace(0.9, 0.5, n), level=np.arange(n, dtype=np.float32), parent=np.array(parent, np.int64),
                link_p=links, junc=np.full((2, 3, 3), heat, np.float16))


def test_union_raw_offsets_parents_and_links():
    x, y = tree(3, [-1, 0, 1], [(1, 2, 0.6)], 0.2), tree(2, [-1, 0], [(0, 1, 0.9)], 0.7)
    y["junc"][0, 0, 0] = 0.1
    u = PI.union_raw([x, y])
    assert u["parent"].tolist() == [-1, 0, 1, -1, 3] and u["link_p"] == [(1, 2, 0.6), (3, 4, 0.9)]
    assert np.array_equal(u["pos"], np.r_[x["pos"], y["pos"]]) and u["junc"].dtype == np.float16
    assert u["junc"][0, 0, 0] == np.float16(0.2) and u["junc"][1, 2, 2] == np.float16(0.7), "junction heat = element-wise maximum"
    try:
        PI.union_raw([x, {k: v for k, v in y.items() if k != "junc"}])
        raise AssertionError("trees with different fields must be refused")
    except ValueError:
        pass


def same_graph(a, b):
    return a["edges"] == b["edges"] and len(a["pts"]) == len(b["pts"]) and all(np.array_equal(p, q) for p, q in zip(a["pts"], b["pts"]))


def test_identity_union_reproduces_the_member():
    """a registry entry of kind union with one member twice gives that member's graphs on every VAL frame; the members are read by their own
    entries (never by the union entry: a run-keyed cache would return the first member's raw output)"""
    import evaluate as EV
    reg, name = EV.registry(), "e24_j1rc_s0"
    if not any(os.path.exists(p) for p in EV.cache_paths(reg[name], "val")):
        print("SKIP test_identity_union_reproduces_the_member: no network-output cache on this machine")
        return
    raw, gts, src = EV.builder_raw(reg[name], "val", require_cache=True)
    calls, real = [], EV.builder_raw
    EV.builder_raw = lambda e, *a, **k: (calls.append(e["run"]), (raw, gts, src))[1]
    try:
        g1, _, _ = EV.graphs_of(name, reg[name], "val", require_cache=True)
        u = dict(kind="union", members=[name, name], tau=reg[name]["tau"], asm=reg[name]["asm"])
        g2, _, info = EV.graphs_of("u", u, "val", require_cache=True)
    finally:
        EV.builder_raw = real
    assert calls == [reg[name]["run"]] * 3 and info["cache"] == [src, src]
    diff = [k for k, (a, b) in enumerate(zip(g1, g2)) if not same_graph(a, b)]
    assert len(g2) == len(raw) == 873 and not diff, f"{len(diff)} of {len(raw)} frames differ, first {diff[:5]}"


if __name__ == "__main__":
    n = 0
    for k, f in sorted(globals().items()):
        if k.startswith("test_") and callable(f):
            f()
            n += 1
    print(f"test_seed_ring_union: {n} tests passed")
