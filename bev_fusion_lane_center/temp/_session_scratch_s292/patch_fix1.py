"""E292 code critic findings 1-4, 6, 7 (scripted edits of the LIVE files, written as .new_s292 and moved over after the tests of the caller). usage (problem folder): python3 temp/_session_scratch_s292/patch_fix1.py"""
import os


def sub(path, pairs):
    s = open(path, encoding="utf-8").read()
    for a, b in pairs:
        assert s.count(a) == 1, (path, a[:90], s.count(a))
        s = s.replace(a, b)
    open(path + ".new_s292", "w", encoding="utf-8", newline="\n").write(s)


# ---- finding 1: startable accepts phase 9 and the gate keys, checks the layer arguments
sub("training/startable.py", [
    ("2 <= x <= 7 for x in v) else \"a non-empty list of phases 2..7\", \"phases to run in order (default 2 3 4 5 6 7)\"),",
     "2 <= x <= 9 for x in v) else \"a non-empty list of phases 2..9\", \"phases to run in order (default 2 3 4 5 6 7; 8 = the end head, 9 = the E292 fine-tune with a frozen pyramid)\"),"),
    ("        \"gate_phases\": (list, None, \"only a failed gate of these phases stops the run\"),\n",
     "        \"gate_phases\": (list, None, \"only a failed gate of these phases stops the run\"),\n"
     "        \"gate_node_prec\": ((int, float), lambda v: None if 0 <= v <= 1 else \"a fraction 0..1\", \"E274: the node gate also needs this node precision within 2 m\"),\n"
     "        \"gate_node_rec\": ((int, float), lambda v: None if 0 <= v <= 1 else \"a fraction 0..1\", \"E274: the node gate also needs this node recall within 2 m\"),\n"),
    ("        \"soft_nms_mnms\": (list, lambda v: None if len(v) == 2 else \"two values: radius m, sigma\", \"E292: the one-pass Matrix-NMS layer after the node head\"),\n",
     "        \"soft_nms_mnms\": (list, lambda v: None if len(v) == 2 and all(isinstance(x, (int, float)) for x in v) and 0 < v[0] <= 2 and v[1] > 0 else \"two numbers: radius m in (0, 2], sigma > 0\", \"E292: the one-pass Matrix-NMS layer after the node head\"),\n"),
    ("({\"phases\": [1]}, \"phases 2..7\")", "({\"phases\": [1]}, \"phases 2..9\"), ({\"phases\": [3, 10]}, \"phases 2..9\"), ({\"soft_nms_mnms\": [2.5, 0.1]}, \"radius m in (0, 2]\"), ({\"soft_nms_mnms\": [\"a\", 0.1]}, \"two numbers\")"),
])
sub("training/train/run_config.py", [
    ("\"resume_from\", \"gate_phases\", ", "\"resume_from\", \"gate_phases\", \"gate_node_prec\", \"gate_node_rec\", "),
])

# ---- finding 4b (CLI chain) and the docstring: one function resolves the two forms of the switch
sub("training/train/qgraph_train.py", [
    ("def main():\n    ap = argparse.ArgumentParser()\n",
     "def resolve_soft_nms(soft_nms, mnms):\n"
     "    \"\"\"the `soft_nms` setting of the model from the two flags: the E277 triple (--soft-nms), the Matrix-NMS pair (--soft-nms-mnms R SIGMA -> ['mnms', R, sigma], the form QGraph and `qpredict.load_model` read from\n"
     "    args['soft_nms']), or None; the two forms together are refused\"\"\"\n"
     "    assert not (soft_nms and mnms), \"--soft-nms and --soft-nms-mnms are two forms of the same switch: give one\"\n"
     "    return [\"mnms\"] + [float(v) for v in mnms] if mnms else soft_nms\n\n\n"
     "def main():\n    ap = argparse.ArgumentParser()\n"),
    ("    assert not (a.soft_nms and a.soft_nms_mnms), \"--soft-nms and --soft-nms-mnms are two forms of the same switch: give one\"\n    if a.soft_nms_mnms:\n        a.soft_nms = [\"mnms\"] + list(a.soft_nms_mnms)                                              # the form that QGraph and qpredict.load_model read\n",
     "    a.soft_nms = resolve_soft_nms(a.soft_nms, a.soft_nms_mnms)                                    # the form that QGraph and qpredict.load_model read (travels in args['soft_nms'])\n"),
    ("Phase 2 pixel maps (the 1 m branch",
     "E292: `--phases 3 4 5 6 9 --resume-from runs/<B run>/model.pt` is the fine-tune of a finished B with the pyramid FROZEN (phase 9 = nodes, links, map encoder, assigner jointly at lr 2e-4, no pixel loss; its gate only reports); `--soft-nms-mnms R SIGMA`\n"
     "puts the one-pass Matrix-NMS layer (`qmnms.py`) after the node head (the E277 soft step is `--soft-nms`; one of the two).\n"
     "Phase 2 pixel maps (the 1 m branch"),
])

# ---- finding 7: qgraph.py header and _toy_batch(B)
sub("training/grid/qgraph.py", [
    ("# phase 2 pixels | 3 nodes | 4 links | 5 sections | 6 assign | 7 joint\n", "# phase 2 pixels | 3 nodes | 4 links | 5 sections | 6 assign | 7 joint | 8 end head | 9 joint without the pixel stage (E292)\n"),
    ("    b = LD.collate([item, item])\n", "    b = LD.collate([item] * B)\n"),
])

# ---- finding 6 and the docstring: reserved tags, seed in the paths
sub("evaluation/qends_eval.py", [
    ("def score_target(a, run_dir):\n",
     "RESERVED_TAGS = (\"snap\", \"chord\", \"attach\", \"all\", \"guard\", \"dedup\", \"nogeom\", \"both\")                       # words of `card_name`: a tag with one of them could name the card of record of the run\n\n\n"
     "def score_target(a, run_dir):\n"),
    ("    assert tag is None or re.fullmatch(r\"[a-z0-9]+\", tag), f\"--tag {tag!r}: lower-case letters and digits only (it is part of a file name)\"\n",
     "    assert tag is None or (re.fullmatch(r\"[a-z0-9]+\", tag) and tag not in RESERVED_TAGS and not re.fullmatch(r\"(thr|r)[0-9]*\", tag)), f\"--tag {tag!r}: lower-case letters and digits only, not a word of the card names (it is part of a file name)\"\n"
     "    seed = getattr(a, \"seed\", None)\n"
     "    for what, path, name in ((\"--full\", full, os.path.basename(str(full))), (\"--base\", base, os.path.basename(os.path.dirname(str(base))))):      # the file name of the dump, the run folder of the checkpoint\n"
     "        assert not path or seed is None or re.search(rf\"_s{seed}(?!\\d)\", name), f\"{what} {path}: its name does not carry _s{seed}: the dump and the checkpoint must be of seed {seed}\"\n"),
    ("(`qgraph_readout` close:1.5@0.5, `qgraph_eval.readout`)",
     "(`qgraph_readout` close:1.5@0.5, `qgraph_eval.readout`; E292: `score --full <dump of B'> --base <B' checkpoint> --spec plain@0.5 --tag a` reads the graph of another head, see `score_target`)"),
    ("        assert refused(NS(thr=0.1, snap=True, tag=\"A b\")), \"a tag that is not a file-name word must be refused\"\n",
     "        assert refused(NS(thr=0.1, snap=True, tag=\"A b\")), \"a tag that is not a file-name word must be refused\"\n"
     "        for t_ in (\"snap\", \"chord\", \"attach\", \"thr0\", \"r\"):\n"
     "            assert refused(NS(thr=0.1, snap=True, tag=t_)), f\"the tag {t_!r} could name a card of record\"\n"
     "        assert refused(NS(thr=0.1, snap=True, seed=0, full=\"temp/e292/predfull_e292_a_s1.pkl\", base=\"runs/e292_a_s0/model.pt\", tag=\"a\")), \"a dump of another seed must be refused\"\n"
     "        assert refused(NS(thr=0.1, snap=True, seed=1, full=\"temp/e292/predfull_e292_a_s1.pkl\", base=\"runs/e292_a_s10/model.pt\", tag=\"a\")), \"_s10 is not _s1\"\n"
     "        assert not refused(NS(thr=0.1, snap=True, seed=1, full=\"temp/e292/predfull_e292_a_s1.pkl\", base=\"runs/e292_a_s1/model.pt\", tag=\"a\")), \"the paths of the right seed must pass\"\n"),
])

# ---- findings 2, 3, 4: the tests
sub("training/grid/test_mnms_layer.py", [
    ("# ---- 9. (--real) the kill criterion of the card: the layer reproduces qgraph_readout.matrix_nms on 20 frames of B's full dump\nif \"--real\" in sys.argv:\n    dump = pickle.load(open(os.path.join(ROOT, \"temp\", \"e276\", \"predfull_e273_p_s0.pkl\"), \"rb\"))\n",
     "# ---- 9. the layer is applied with the model's OWN (R, sigma) (at a fresh init nothing is damped, so section 1 alone would pass a model that skipped the layer): on the squeezed model no['e'] is the damped logit of\n"
     "# that setting, it differs from e_raw in many cells, and a model with another sigma delivers other values\n"
     "def node_out(model):\n    with torch.no_grad():\n        return model({\"x\": x}, upto=\"nodes\", gt=False)[\"no\"]\n\n\n"
     "o_sq = node_out(mm_s)\n"
     "check(torch.allclose(o_sq[\"e\"], QMN.matrix_nms_logit(o_sq[\"e_raw\"], o_sq[\"pos\"], 1.75, 0.1), atol=1e-5), \"no['e'] of the squeezed model must be the damped logit of the model's own (R, sigma)\")\n"
     "n_damped = int((o_sq[\"e\"] < o_sq[\"e_raw\"] - 1e-4).sum())\n"
     "check(n_damped > 500, f\"the squeezed model damps only {n_damped} cells: the layer is not applied\")\n"
     "mm_other = QG.QGraph(width=16, f05=16, d=32, soft_nms=[\"mnms\", 1.75, 1.0])\n"
     "mm_other.load_state_dict(plain.state_dict())\n"
     "check(not torch.allclose(node_out(squeezed(mm_other))[\"e\"], o_sq[\"e\"], atol=1e-3), \"another sigma must deliver other values (a hard-wired sigma would not)\")\n\n"
     "# ---- 10. phase 9 = phase 7 without the pixel loss, term by term (one draw of the random part); a phase set that loses a loss term fails here\n"
     "def seeded(model, phase):\n    torch.manual_seed(7)\n    model.zero_grad()\n    loss, info = QG.qgraph_loss(model, b, phase)\n    return float(loss), info\n\n\n"
     "l7, i7 = seeded(mm, 7)\nl2, i2 = seeded(mm, 2)\nl9, i9 = seeded(mm, 9)\n"
     "check(abs(l7 - l2 - l9) < 1e-3 * max(1.0, abs(l9)), (l7, l2, l9))\n"
     "check(set(i9) == set(i7) - set(i2) and all(abs(i9[k] - i7[k]) < 1e-3 * max(1.0, abs(i7[k])) for k in i9), (sorted(i9), sorted(set(i7) - set(i2))))\n"
     "check(any(k.startswith(p) for k in i9 for p in (\"n_\", \"l_\", \"s_\", \"a_\")) and not any(k.startswith(\"px_\") for k in i9) and {k[:2] for k in i9} >= {\"n_\", \"l_\", \"s_\", \"a_\"}, sorted(i9))\n\n"
     "# ---- 11. an optimiser step as qgraph_train.py sets it up (requires_grad toggled, params=trainable_params): the pyramid stays torch.equal, the node head moves\n"
     "mm2 = copy.deepcopy(mm)\n"
     "pyr0 = {k: v.clone() for k, v in mm2.pyr.state_dict().items()}\nnodes0 = {k: v.clone() for k, v in mm2.nodes.state_dict().items()}\n"
     "for p in mm2.parameters():\n    p.requires_grad_(False)\nps9 = QG.trainable_params(mm2, 9)\nfor p in ps9:\n    p.requires_grad_(True)\n"
     "opt = torch.optim.Adam(ps9, lr=1e-3)\n"
     "for _ in range(2):\n    torch.manual_seed(7)\n    loss9, _ = QG.qgraph_loss(mm2, b, 9)\n    opt.zero_grad()\n    loss9.backward()\n    opt.step()\n"
     "check(all(torch.equal(pyr0[k], v) for k, v in mm2.pyr.state_dict().items()), \"the pyramid changed in a phase 9 step\")\n"
     "check(any(not torch.equal(nodes0[k], v) for k, v in mm2.nodes.state_dict().items()), \"the node head did not move in a phase 9 step\")\n\n"
     "# ---- 12. the switch end to end: --soft-nms-mnms -> args['soft_nms'] -> model.pt -> load_model; the two forms together are refused\n"
     "sys.path.insert(0, os.path.join(ROOT, \"training\", \"train\"))\nimport qgraph_train as QT  # noqa: E402\n\n"
     "sn = QT.resolve_soft_nms(None, [1.75, 0.1])\ncheck(sn == [\"mnms\", 1.75, 0.1] and QT.resolve_soft_nms([1.5, 0.1, 3.0], None) == [1.5, 0.1, 3.0] and QT.resolve_soft_nms(None, None) is None, sn)\n"
     "refused(lambda: QT.resolve_soft_nms([1.5, 0.1, 3.0], [1.75, 0.1]), \"both forms of the switch\")\n"
     "with tempfile.TemporaryDirectory() as d:\n"
     "    torch.save(dict(state=QG.QGraph(width=16, f05=16, d=32, soft_nms=sn).state_dict(), args=dict(width=48, soft_nms=sn)), os.path.join(d, \"model.pt\"))\n"
     "    check(QP.load_model(d, \"cpu\").mnms == (1.75, 0.1), \"the chain flag -> args -> model.pt -> load_model must restore the layer\")\n\n"
     "# ---- 13. the gradient of the log-space layer against the analytic one: two nodes 1 m apart, the stronger A (e 0.9) over the weaker B (e 0.7); p = e_B x exp(-o^2 / sigma), o = 1 - d / R\n"
     "zz = torch.logit(torch.tensor([[[0.9, 0.7]]])).requires_grad_(True)\n"
     "pz = pp.clone().requires_grad_(True)\n"
     "QMN.matrix_nms_logit(zz, pz, 1.75, 0.1)[0, 0, 1].backward()\n"
     "dist = float(((pp[0, :, 0, 0] - pp[0, :, 0, 1]) ** 2).sum().sqrt())\n"
     "o_ = 1.0 - dist / 1.75\nd_ = float(np.exp(-o_ ** 2 / 0.1))\np_ = 0.7 * d_\n"
     "check(abs(float(zz.grad[0, 0, 1]) - (1 - 0.7) / (1 - p_)) < 1e-4 and float(zz.grad[0, 0, 0]) == 0.0, (float(zz.grad[0, 0, 1]), (1 - 0.7) / (1 - p_)))        # d logit / d z_B = (1 - e) / (1 - e d); the order is a comparison: nothing for A\n"
     "dpos = (2 * o_ / 0.1) / (1 - p_) / 1.75\n"
     "check(abs(float(pz.grad[0, 1, 0, 1]) - dpos) < 1e-3 * dpos and abs(float(pz.grad[0, 1, 0, 0]) + dpos) < 1e-3 * dpos, (float(pz.grad[0, 1, 0, 1]), dpos))        # moving B away along x lifts e'; moving A away does the same\n\n"
     "# ---- 14. (--real) the kill criterion of the card: the layer reproduces qgraph_readout.matrix_nms on 20 frames of B's full dump\n"
     "if \"--real\" in sys.argv:\n    dump_path = os.path.join(ROOT, \"temp\", \"e276\", \"predfull_e273_p_s0.pkl\")\n"
     "    assert os.path.exists(dump_path), f\"--real needs B's full dump {dump_path} (qgraph_run_eval.py score --run e273_p_s0 --seed 0 --exp e276 --readouts plain@0.5 --dump-full)\"\n"
     "    dump = pickle.load(open(dump_path, \"rb\"))\n"),
])
print("patched")
