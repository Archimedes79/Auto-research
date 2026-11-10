"""Spec tests of evaluation/qgraph_eval.py, the local scorer of the lane head (E273, card section 4), on REAL VAL frames. CPU only (about 4 minutes).

  * frames: the 873 VAL frames are the rows of BuilderData('val_clean'), their tokens (val_tokens.json[row]) are the keys of the lane GT store, and the lane GT of a token lies on the old graph
    GT of its row (so the token mapping is right); the share of the VAL tokens that is stored is printed;
  * oracle: the lane GT through the whole scorer (68 frames): node F1 1, link P / R 1, F1@1.0 against the old graph GT >= 0.99, lane_acc 1, one section per GT piece, no id switch;
  * readout rules: a piece lies in one section, every kept link between two sections is a connector piece, edges join end node to start node, thresholds decide what is delivered;
  * degradations move the numbers by a predictable amount: nodes shifted (position error = the shift, F1 at the tolerance falls), 10 % of the nodes dropped (node recall 0.9), 20 % of the links cut
    (link recall 0.8, topo recall falls), the section ids of two pieces swapped half by half (known counts of ids and switches; a pure relabelling changes nothing), 20 % of the assignment
    randomised (lane_acc falls by the chance-corrected 0.2), all-dustbin (lane_acc 0), nodes far from every lane (dustbin precision / recall), an empty prediction (0 without a crash);
  * reproduction: the stored graphs of recall_sp0_dp scored through `graph_metrics` give the numbers of the lane suite run graphs_recall_sp0_dp_carry_20261104_181557 (per frame counts and
    scores identical on 40 frames; pooled over all 873: Q 0.7424, F1@1.0 0.8781, topo F1 0.7405, junction F1 0.2730 to 4 digits).
Run (from the problem folder): wsl bash tools/env/_py14.sh test_qgraph_eval.py [word ...]     (add CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 on the shared machine)
"""
import os
import sys
import time
import traceback

import numpy as np

import bflc_paths as BP
import lane_suite as LS
import qformat
import qgraph_eval as QE

_CACHE = {}


def frames(step=13):
    """every `step`-th VAL frame (68 for 13) with its lane GT, and the oracle prediction of each"""
    if ("f", step) not in _CACHE:
        fr = QE.val_frames()[::step]
        _CACHE[("f", step)] = (fr, [qformat.oracle_prediction(f.lane()) for f in fr])
    return _CACHE[("f", step)]


def clone(p):
    c = dict(nodes={k: np.array(v) for k, v in p["nodes"].items()}, links={k: ([np.array(x) for x in v] if k == "curves" and v is not None else np.array(v) if v is not None else None) for k, v in p["links"].items()})
    a = p.get("assign")
    c["assign"] = None if a is None else dict(P=a["P"].copy(), map_ids=list(a["map_ids"]), road=a["road"].copy(), road_ids=list(a["road_ids"]))
    return c


def show(name, d, keys):
    print(f"   {name:34s}" + "  ".join(f"{k} {d[k]:.4f}" for k in keys))


def score(preds, fr, **kw):
    return QE.score_frames(preds, fr, **kw)


# ------------------------------------------------------------------------------------------------------------------------------ frames
def test_frames_tokens_and_store():
    import p16_data as PD
    fr = QE.val_frames()
    assert len(fr) == 873 and [f.row for f in fr] == [int(r) for r in PD.BuilderData("val_clean").rows], "the VAL frames are the rows of BuilderData('val_clean')"
    assert len({f.token for f in fr}) == 873
    share = QE.stored_share(fr)
    print(f"   share of the 873 VAL tokens with a stored lane GT: {share:.4f} (signature of this checkout's GtGenerator source)")
    assert share > 0, "no lane GT stored for the VAL tokens"
    from scipy.spatial import cKDTree
    hit = []
    for f in fr[::29]:
        if f.token not in _held():
            continue
        P = np.concatenate([np.asarray(p["pts"]) for p in f.lane()["pieces"]])
        G = np.concatenate([np.asarray(p) for p in f.gt["pts"]])
        P, G = P[(np.abs(P) < 48).all(1)], G[(np.abs(G) < 48).all(1)]
        hit.append(((cKDTree(P).query(G)[0] < 0.3).mean(), (cKDTree(G).query(P)[0] < 0.3).mean()))
    hit = np.array(hit)
    print(f"   old graph GT points within 0.3 m of the lane GT of the token: {hit[:, 0].mean():.4f}, the other way: {hit[:, 1].mean():.4f} ({len(hit)} frames)")
    assert hit.min() > 0.95, "the token of a row is not the sample of its graph GT"


def _held():
    if "held" not in _CACHE:
        sys.path.insert(0, BP.REPO)
        from GtGenerator import lane_gt
        _CACHE["held"] = lane_gt.held(os.path.join(lane_gt.DATA, lane_gt.signature()))
    return _CACHE["held"]


# ------------------------------------------------------------------------------------------------------------------------------ oracle
def test_oracle():
    fr, preds = frames()
    assert len(fr) >= 60
    out, rec = score(preds, fr, records=True)
    g, n, a = out["graph"], out["nodes"], out["assign"]
    show("graph", g, ("F1@1.0", "Q", "lanes_found", "topo_F1", "junction_F1", "dup_frac", "R@1_far", "vert100"))
    show("nodes", n, ("node_F1", "pos_err_med", "link_short_P", "link_short_R", "link_long_P", "link_long_R", "degree_acc", "nodes_per_100m"))
    show("assign + sections", dict(a, **g), ("lane_acc", "purity", "road_acc", "pieces_per_gt_piece", "id_switches_per_100m"))
    assert n["node_F1"] == 1.0 and n["pos_err_med"] == 0.0 and n["dir_err_med"] == 0.0
    assert all(n[f"link_{t}_{x}"] == 1.0 for t in ("short", "long") for x in "PR") and n["degree_acc"] == 1.0
    assert all(n[f"ev_{f}_P"] == 1.0 and n[f"ev_{f}_R"] == 1.0 for f in ("stop", "exit", "split", "merge", "start", "end", "win_in", "win_out"))
    assert g["F1@1.0"] >= 0.99 and g["P@1.0"] > 0.999 and g["false_len_frac"] < 0.001, "the oracle graph must lie on the old graph GT"
    assert g["topo_F1"] > 0.95 and g["junction_F1"] > 0.9 and g["R@1_far"] > 0.97, "...and carry its connectivity (the representation's own ceiling against the old GT)"
    assert a["lane_acc"] == 1.0 and a["purity"] == 1.0 and a["road_acc"] == 1.0
    assert g["pieces_per_gt_piece"] == 1.0 and g["id_switches_per_100m"] == 0.0
    assert out["lane_missing"] == 0
    # scoring one frame alone and pooling the counts is the same as score_frames
    one = QE.graph_metrics(QE.readout(preds[3])[0], fr[3])
    assert abs(one["F1@1.0"] - QE.graph_numbers([rec["graph"][3]])["F1@1.0"]) < 1e-12


# ------------------------------------------------------------------------------------------------------------------------------ degradations
def shifted(preds, d):
    out = []
    for p in preds:
        q = clone(p)
        q["nodes"]["pos"] = q["nodes"]["pos"] + [d, 0.0]
        q["links"]["curves"] = [c + [d, 0.0] for c in q["links"]["curves"]]
        out.append(q)
    return out


def test_shift_nodes():
    fr, preds = frames()
    base = score(preds, fr)
    res = {}
    for d in (0.5, 1.0, 1.5):                                              # across the lanes (they run along y); beyond 2 m a node matches the next lane
        res[d] = score(shifted(preds, d), fr)
        show(f"nodes shifted {d} m (across)", dict(res[d]["nodes"], **res[d]["graph"]), ("pos_err_med", "node_F1", "F1@0.5", "F1@1.0", "F1@1.5", "topo_F1"))
        assert abs(res[d]["nodes"]["pos_err_med"] - d) < 1e-9, "the position error of matched nodes is the shift"
    g0 = base["graph"]
    assert res[0.5]["graph"]["F1@1.0"] > g0["F1@1.0"] - 0.01 and res[0.5]["graph"]["F1@0.5"] < g0["F1@0.5"] - 0.3, "0.5 m: inside 1.0 m, outside 0.5 m"
    f = [g0["F1@1.0"]] + [res[d]["graph"]["F1@1.0"] for d in (0.5, 1.0, 1.5)]
    assert all(a >= b - 1e-9 for a, b in zip(f[:-1], f[1:])) and f[2] < f[0] - 0.25 and f[3] < f[2] - 0.1, f
    assert res[0.5]["nodes"]["node_F1"] > 0.99 and res[1.0]["nodes"]["node_F1"] > 0.9, "node F1 (tolerance 2 m) holds up to 1 m"
    assert res[1.0]["graph"]["topo_F1"] < g0["topo_F1"] - 0.2


def test_readout_structure():
    """the rules of the output step: a piece lies in ONE section, every kept link between two sections is a connector piece, an edge joins the end node of a piece to the start node of the next,
    thresholds and kept nodes decide what is delivered"""
    fr, preds = frames(step=43)
    n_link = n_edge = 0
    for p in preds:
        graph, bk = QE.readout(p)
        sec, src, dst = p["nodes"]["section"], p["links"]["src"], p["links"]["dst"]
        L = bk["links"]
        assert len(L) == len(src), "the oracle keeps every link"
        cross = sec[src] != sec[dst]
        assert sum(k == "link" for k in bk["kind"]) == cross.sum() and list(bk["cross"]) == list(cross[L])
        n_link, n_edge = n_link + cross.sum(), n_edge + len(graph.edges)
        for k, n, s_ in zip(bk["kind"], bk["piece_nodes"], bk["section"]):
            if k == "section":
                assert set(sec[n]) == {s_}, "a section piece holds one section"
        n_in = sum(len(l) for k, l in zip(bk["kind"], bk["piece_links"]) if k == "section")
        assert n_in == (~cross).sum(), "every link inside a section is on a section piece"
        assert all(graph.nodes[i][-1] == graph.nodes[j][0] for i, j in graph.edges), "an edge goes from the end node of a piece to the start node of the next"
        assert all(len(set(map(tuple, np.round(l, 6))) ) > 1 for l in graph.lines), "no piece of zero length"
        # thresholds: a link below thr_link or an end node below thr_e is not delivered
        q = clone(p)
        q["links"]["p"][:] = 0.4
        assert len(QE.readout(q)[0].lines) == 0 and len(QE.readout(q, thr_link=0.3)[0].lines) == len(graph.lines)
        q = clone(p)
        q["nodes"]["e"][:] = 0.4
        assert len(QE.readout(q)[0].lines) == 0 and len(QE.readout(q, thr_e=0.3)[0].lines) == len(graph.lines)
    assert n_link > 0 and n_edge > 0, "the frames hold connectors and edges"


def test_hermite_path_flags_and_directions():
    fr, preds = frames()
    base = score(preds, fr)
    ps, pf, pd = [], [], []
    for p in preds:
        q = clone(p)
        q["links"]["curves"] = None                                        # no stored curves: the Hermite curves of the node poses and the handles (1/3, 1/3)
        ps.append(q)
        q = clone(p)
        q["nodes"]["flags"][:, QE.QT.FI["split"]] = 0.0                    # no split flag anywhere
        pf.append(q)
        q = clone(p)
        c, s_ = np.cos(np.radians(20)), np.sin(np.radians(20))
        q["nodes"]["dirv"] = q["nodes"]["dirv"] @ np.array([[c, s_], [-s_, c]])   # every direction turned by 20 degrees
        pd.append(q)
    h, f, d = score(ps, fr), score(pf, fr), score(pd, fr)
    show("oracle / Hermite handles 1/3", dict(h["graph"], f1_oracle=base["graph"]["F1@1.0"]), ("f1_oracle", "F1@1.0", "topo_F1", "junction_F1"))
    assert h["graph"]["F1@1.0"] > base["graph"]["F1@1.0"] - 0.04 and h["graph"]["topo_F1"] > base["graph"]["topo_F1"] - 0.08, "chord / 3 handles nearly reproduce the curves (the long links want learned scales)"
    assert f["nodes"]["ev_split_R"] == 0.0 and f["nodes"]["ev_split_P"] == 0.0 and f["nodes"]["ev_merge_R"] == 1.0 and f["nodes"]["node_F1"] == 1.0
    assert abs(d["nodes"]["dir_err_med"] - 20.0) < 1e-6 and d["nodes"]["pos_err_med"] == 0.0


def test_drop_nodes_and_cut_links():
    fr, preds = frames()
    base = score(preds, fr)
    rng = np.random.RandomState(0)
    pd, pc = [], []
    for p in preds:
        q = clone(p)
        q["nodes"]["e"] = np.where(rng.rand(len(q["nodes"]["e"])) < 0.10, 0.0, 1.0)
        pd.append(q)
        q = clone(p)
        q["links"]["p"] = np.where(rng.rand(len(q["links"]["p"])) < 0.20, 0.0, 1.0)
        pc.append(q)
    d, c = score(pd, fr), score(pc, fr)
    show("10 % of the nodes dropped", dict(d["nodes"], **d["graph"]), ("node_R", "node_P", "R@1.0", "topo_R", "topo_F1", "lanes_found"))
    show("20 % of the links cut", dict(c["nodes"], **c["graph"]), ("link_short_R", "link_short_P", "R@1.0", "topo_R", "topo_F1", "lanes_found"))
    assert abs(d["nodes"]["node_R"] - 0.9) < 0.02 and d["nodes"]["node_P"] == 1.0
    assert d["graph"]["R@1.0"] < base["graph"]["R@1.0"] - 0.05 and d["graph"]["topo_F1"] < base["graph"]["topo_F1"] - 0.05
    assert abs(c["nodes"]["link_short_R"] - 0.8) < 0.03 and c["nodes"]["link_short_P"] == 1.0
    assert c["graph"]["topo_R"] < base["graph"]["topo_R"] - 0.05 and c["graph"]["topo_F1"] < base["graph"]["topo_F1"] - 0.05, "cut links are a connectivity error"


def test_sections_swap_and_relabel():
    fr, preds = frames()
    swapped, relabelled, n_swapped = [], [], 0
    for p in preds:
        nt = p["table"]
        q = clone(p)
        sec = q["nodes"]["section"].copy()
        big = [i for i in np.unique(nt["piece"]) if (nt["piece"] == i).sum() >= 12]
        if len(big) >= 2:                                                  # swap the section ids of the second halves of two pieces
            a, b = big[:2]
            halves, h = [], min((nt["piece"] == a).sum(), (nt["piece"] == b).sum()) // 2
            for i in (a, b):
                m = np.flatnonzero(nt["piece"] == i)
                halves.append(m[np.argsort(nt["j"][m])][-h:])
            sec[halves[0]], sec[halves[1]] = p["nodes"]["section"][halves[1]], p["nodes"]["section"][halves[0]]
            n_swapped += 1
        q["nodes"]["section"] = sec
        swapped.append(q)
        r = clone(p)
        r["nodes"]["section"] = r["nodes"]["section"] * 7 + 100               # an injective relabelling
        relabelled.append(r)
    base, sw, rl = (score(x, fr, records=True) for x in (preds, swapped, relabelled))
    show("oracle", base[0]["graph"], ("pieces_per_gt_piece", "id_switches_per_100m", "F1@1.0", "topo_F1"))
    show("2 pieces' second halves swapped", sw[0]["graph"], ("pieces_per_gt_piece", "id_switches_per_100m", "F1@1.0", "topo_F1"))
    sec = lambda rec: np.sum([c["sec"] for c in rec["graph"]], 0)               # distinct sections, switches, GT pieces with a node, lane length
    s0, s1 = sec(base[1]), sec(sw[1])
    assert s1[1] - s0[1] == 2 * n_swapped and s1[0] - s0[0] == 2 * n_swapped and s1[2] == s0[2], (s0, s1, n_swapped)     # each swapped piece has 2 ids and 1 switch
    assert sw[0]["graph"]["pieces_per_gt_piece"] > 1.0 and sw[0]["graph"]["id_switches_per_100m"] > 0.0
    assert abs(sw[0]["graph"]["F1@1.0"] - base[0]["graph"]["F1@1.0"]) < 0.01, "the geometry does not depend on the ids"
    for grp in ("graph", "nodes", "assign"):                                    # a pure relabelling changes nothing (NaN == NaN here)
        for k, v in base[0][grp].items():
            w = rl[0][grp][k]
            assert w == v or (v != v and w != w), (grp, k, v, w)


def test_assignment_degradations():
    fr, preds = frames()
    rng = np.random.RandomState(1)
    rnd, dust, junk_bin, junk_piece, expect, n_all = [], [], [], [], 0.0, 0
    for f, p in zip(fr, preds):
        K = len(p["assign"]["map_ids"])
        q = clone(p)
        M = len(q["nodes"]["e"])
        hit = rng.rand(M) < 0.20
        P = q["assign"]["P"]
        P[hit] = np.eye(K + 1)[rng.randint(0, K + 1, hit.sum())]                 # a random one-hot row for 20 % of the nodes
        expect += 0.20 * M * (1 - 1.0 / (K + 1))
        n_all += M
        rnd.append(q)
        q = clone(p)
        q["assign"]["P"][:] = np.eye(K + 1)[K]                                    # everything to the dustbin
        dust.append(q)
        # 15 junk nodes far from every lane, assigned to the dustbin / to piece 0
        ref = f.ref()
        far = rng.uniform(-45, 45, (400, 2))
        far = far[ref.near(far)[0] > 4.0][:15]
        for lst, col in ((junk_bin, K), (junk_piece, 0)):
            q = clone(p)
            n = len(far)
            q["nodes"] = {k: np.concatenate([v, np.zeros((n,) + v.shape[1:])]) for k, v in q["nodes"].items()}
            q["nodes"]["pos"][-n:] = far
            q["nodes"]["e"][-n:] = 1.0
            q["nodes"]["dirv"][-n:] = [0.0, 1.0]
            q["nodes"]["section"][-n:] = 10 ** 6
            q["assign"]["P"] = np.concatenate([q["assign"]["P"], np.tile(np.eye(K + 1)[col], (n, 1))])
            q["assign"]["road"] = np.concatenate([q["assign"]["road"], np.tile(np.eye(q["assign"]["road"].shape[1])[0], (n, 1))])
            q["links"]["src"], q["links"]["dst"] = q["links"]["src"].astype(int), q["links"]["dst"].astype(int)
            lst.append(q)
    r, d = score(rnd, fr)["assign"], score(dust, fr)["assign"]
    jb, jp = score(junk_bin, fr)["assign"], score(junk_piece, fr)["assign"]
    show("20 % of the assignment random", r, ("lane_acc", "purity", "road_acc"))
    show("all dustbin", d, ("lane_acc", "dust_P", "purity"))
    show("far nodes -> dustbin / -> piece 0", dict(jb, jp_P=jp["dust_P"], jp_R=jp["dust_R"], jb_P=jb["dust_P"], jb_R=jb["dust_R"]), ("jb_P", "jb_R", "jp_P", "jp_R"))
    assert abs((1.0 - r["lane_acc"]) - expect / n_all) < 0.02, (1.0 - r["lane_acc"], expect / n_all)
    assert d["lane_acc"] == 0.0 and d["dust_P"] == 0.0 and d["purity"] == 1.0, "purity is consistency: a lane wholly on one (wrong) piece is pure"
    assert r["purity"] < 1.0 - 0.1
    assert jb["dust_P"] == 1.0 and jb["dust_R"] == 1.0 and jb["lane_acc"] == 1.0, "far nodes in the dustbin are right"
    assert jp["dust_P"] != jp["dust_P"] or jp["dust_R"] == 0.0, "far nodes on a piece are a missed dustbin"
    assert jp["lane_acc"] == 1.0


def test_empty_prediction():
    fr, preds = frames()
    ps = []
    for p in preds:
        q = clone(p)
        K = len(q["assign"]["map_ids"])
        q["nodes"] = {k: v[:0] for k, v in q["nodes"].items()}
        q["links"] = {k: (v[:0] if v is not None and k != "curves" else ([] if v is not None else None)) for k, v in q["links"].items()}
        q["assign"] = dict(P=np.zeros((0, K + 1)), map_ids=q["assign"]["map_ids"], road=np.zeros((0, q["assign"]["road"].shape[1])), road_ids=q["assign"]["road_ids"])
        ps.append(q)
    out = score(ps, fr)
    show("empty prediction", dict(out["nodes"], **out["graph"], **out["assign"]), ("F1@1.0", "lanes_found", "topo_F1", "junction_F1", "node_F1", "lane_acc", "Q"))
    g = out["graph"]
    assert g["F1@1.0"] == 0.0 and g["lanes_found"] == 0.0 and g["topo_F1"] == 0.0 and g["junction_F1"] == 0.0 and g["R@1_far"] == 0.0
    assert out["nodes"]["node_F1"] == 0.0 and out["assign"]["lane_acc"] == 0.0
    assert abs(g["Q"] - 0.15 / 0.85) < 1e-9, "Q of an empty graph is only its (1 - false length) term: the project's definition"


def test_paired_bootstrap():
    fr, preds = frames()
    shifted = []
    for p in preds:
        q = clone(p)
        q["nodes"]["pos"] = q["nodes"]["pos"] + [1.5, 0.0]
        q["links"]["curves"] = [c + [1.5, 0.0] for c in q["links"]["curves"]]
        shifted.append(q)
    (_, ra), (_, rb) = score(preds, fr, records=True), score(shifted, fr, records=True)
    pr = QE.paired(ra, rb, fr, B=30)
    print(f"   oracle - shifted 1.5 m: F1@1.0 {pr['F1@1.0']['diff']:.3f} [{pr['F1@1.0']['lo']:.3f}, {pr['F1@1.0']['hi']:.3f}], topo_F1 {pr['topo_F1']['diff']:.3f}")
    assert pr["F1@1.0"]["lo"] > 0.1 and "topo_F1" in pr


# ------------------------------------------------------------------------------------------------------------------------------ reproduction
RUN = f"{BP.ROOT}/runs/suite/graphs_recall_sp0_dp/quant/graphs_recall_sp0_dp_carry_20261104_181557"


def test_reproduce_lane_suite_run():
    sys.path.insert(0, BP.REPO + "/autoresearch")
    from Evaluation.store import RunStore
    from lane_task import LaneTopoMetric
    st = RunStore.open(RUN)
    sub = LS.GraphSubject("recall_sp0_dp")                                 # the stored graphs of d, assembled on the CPU as the suite does
    fr = QE.val_frames()
    pos = {fr_["key"]: i for i, fr_ in enumerate(st.frames)}
    lcr, lcr_topo, topo = st.counts("lcr"), st.counts("lcr_topo"), st.counts("topo")
    cs = [QE.graph_counts(sub.graphs[f.row], f) for f in fr]               # all 873 frames
    # (1) per-frame: the additive counts are the ones the suite stored
    worst = 0.0
    for k in range(0, 873, 22):                                            # 40 frames
        f, i = fr[k], pos[f"row{fr[k].row}"]
        c = cs[k]
        mine = QE.graph_numbers([c])
        sc = st.frames[i]["scores"]
        for a, b in ((mine["F1@1.0"], sc["lcr.F1@1.0"]), (mine["topo_F1"], sc["lcr_topo.topo_F1"])):
            worst = max(worst, abs(a - b))
        assert np.array_equal(c["lane"]["marb"], topo[i]["marb"]) and np.array_equal(c["lane"]["junc"], topo[i]["junc"]), "lane_task topo counts differ from the stored ones"
        assert np.array_equal(c["topo"]["topo"], lcr_topo[i]["topo"]) and np.array_equal(c["topo"]["dup"], lcr_topo[i]["dup"]) and np.array_equal(c["topo"]["rng"], lcr_topo[i]["rng"])
        rs = c["rec"]["rs"]
        assert np.array_equal(rs["cnt"], lcr[i]["cnt"]) and np.array_equal(rs["m1"], lcr[i]["m1"]) and rs["n_p"] == lcr[i]["misc"][0] and c["rec"]["nv"] == lcr[i]["misc"][8]
    print(f"   40 frames: counts identical to the stored counts, per-frame F1@1.0 and topo F1 differ by at most {worst:.1e}")
    assert worst < 1e-9
    # (2) pooled over the 873 frames: the card's numbers
    mine = QE.graph_numbers(cs)
    suite = LS.LcrMetric().reduce(lcr)
    suite.update(LS.LcrTopoMetric().reduce(lcr_topo))
    suite.update({k: v for k, v in LaneTopoMetric(LS.config()).reduce(topo).items() if k.startswith("junction")})
    keys = ("Q", "S", "J", "F1@1.0", "lanes_found", "vert100", "topo_F1", "topo_P", "topo_R", "dup_frac", "R@1_far", "junction_F1", "junction_P", "junction_R")
    print("   873 frames, mine / suite run:", "  ".join(f"{k} {mine[k]:.4f}/{suite[k]:.4f}" for k in keys))
    for k in keys:
        assert abs(mine[k] - suite[k]) < 1e-9, (k, mine[k], suite[k])
    for k, v in (("Q", 0.7424), ("F1@1.0", 0.8781), ("topo_F1", 0.7405), ("junction_F1", 0.2730), ("junction_P", 0.2125), ("junction_R", 0.3815), ("lanes_found", 0.7809), ("dup_frac", 0.0766)):
        assert abs(mine[k] - v) < 5e-5, (k, mine[k], v)
    # (3) the same through the pooled path of score_frames (raw graph dicts) on the 40 frames
    sel = fr[::22]
    out = QE.score_frames([sub.graphs[f.row] for f in sel], sel)["graph"]
    idx = [pos[f"row{f.row}"] for f in sel]
    ref = LS.LcrMetric().reduce([lcr[i] for i in idx])
    assert abs(out["Q"] - ref["Q"]) < 1e-9 and abs(out["F1@1.0"] - ref["F1@1.0"]) < 1e-9


CHECKS = [("frames, tokens and the store", test_frames_tokens_and_store), ("oracle through the scorer", test_oracle), ("readout structure", test_readout_structure), ("nodes shifted", test_shift_nodes),
          ("Hermite path, flags, directions", test_hermite_path_flags_and_directions),
          ("nodes dropped, links cut", test_drop_nodes_and_cut_links), ("sections swapped / relabelled", test_sections_swap_and_relabel),
          ("assignment degraded", test_assignment_degradations), ("empty prediction", test_empty_prediction), ("paired bootstrap", test_paired_bootstrap),
          ("reproduce the lane suite run of recall_sp0_dp", test_reproduce_lane_suite_run), ("toy self-test", QE._selftest)]


def main(only=()) -> int:
    t0, bad = time.time(), 0
    checks = [c for c in CHECKS if not only or any(o in c[0] for o in only)]
    for name, fn in checks:
        t = time.time()
        print(f"... {name}")
        try:
            fn()
            print(f"ok    {name}  ({time.time() - t:.1f} s)")
        except Exception:                                                  # noqa: BLE001
            bad += 1
            print(f"FAIL  {name}\n{traceback.format_exc()}")
    print(f"qgraph_eval tests: {len(checks) - bad}/{len(checks)} checks passed in {time.time() - t0:.0f} s")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
