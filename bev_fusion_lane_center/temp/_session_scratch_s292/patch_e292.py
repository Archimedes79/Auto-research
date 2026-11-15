"""E292 code step (scratch, scripted text surgery on COPIES in temp/_session_scratch_s292/new/): phase 9 (fine-tune without the pixel stage, pyramid frozen), the Matrix-NMS form of the soft
close-node step QGraph(soft_nms=('mnms', R, sigma)), the gate entry (numbers only), the CLI flag, the run_config / startable keys. The live files are not touched; install = atomic mv after the tests.
usage (from the problem folder): python temp/_session_scratch_s292/patch_e292.py"""
import os
import shutil

NEW = "temp/_session_scratch_s292/new"
os.makedirs(NEW, exist_ok=True)


def sub(path, pairs):
    dst = os.path.join(NEW, os.path.basename(path))
    s = open(path, encoding="utf-8").read()
    for a, b in pairs:
        assert s.count(a) == 1, (path, a[:80], s.count(a))
        s = s.replace(a, b)
    open(dst, "w", encoding="utf-8", newline="\n").write(s)
    return dst


# ---- qgraph.py
sub("training/grid/qgraph.py", [
    ("import qlinks as QL\nimport qnodes as QN\n", "import qlinks as QL\nimport qmnms as QMN\nimport qnodes as QN\n"),
    ('PHASE_UPTO = {2: "pyr", 3: "nodes", 4: "links", 5: "sections", 6: "assign", 7: "assign"}', 'PHASE_UPTO = {2: "pyr", 3: "nodes", 4: "links", 5: "sections", 6: "assign", 7: "assign", 9: "assign"}'),
    ("PHASE_FROZEN[8] = (\"pyr\", \"nodes\")\n",
     "PHASE_FROZEN[8] = (\"pyr\", \"nodes\")\n"
     "# E292: phase 9 is the joint phase 7 without the pixel stage: nodes, links, map encoder and assigner train together on a FROZEN pyramid (the fine-tune of B's final checkpoint, so E312's maps and pair head stay valid)\n"
     "PHASE_TRAIN[9] = (\"nodes\", \"links\", \"menc\", \"asg\")\n"
     "PHASE_FROZEN[9] = (\"pyr\",)\n"),
    ("    def __init__(self, width=48, f05=32, d=64, soft_nms=None, ends=False):\n"
     "        \"\"\"soft_nms = None (the E273 head) or (radius m, tau m, passes) of `qnodes.soft_suppress_logit` (E277: the close-node suppression as a soft step right after the node head: the existence\n"
     "        logit `no['e']` that every later stage, the loss and the read-out see is the damped one; the head's own logit stays in `no['e_raw']`). A property of the model, not of the loss: it travels\n"
     "        with the checkpoint (`args['soft_nms']`, read by `qpredict.load_model`).\"\"\"\n"
     "        super().__init__()\n"
     "        self.soft_nms = tuple(float(v) for v in soft_nms) if soft_nms else None\n",
     "    def __init__(self, width=48, f05=32, d=64, soft_nms=None, ends=False):\n"
     "        \"\"\"soft_nms = None (the E273 head) or (radius m, tau m, passes) of `qnodes.soft_suppress_logit` (E277: the close-node suppression as a soft step right after the node head: the existence\n"
     "        logit `no['e']` that every later stage, the loss and the read-out see is the damped one; the head's own logit stays in `no['e_raw']`) or ('mnms', radius m, sigma) of `qmnms.matrix_nms_logit`\n"
     "        (E292: the one-pass Matrix-NMS decay in the same place, with a hard neighbour ORDER and a smooth decision to drop; no parameters). A property of the model, not of the loss: it travels\n"
     "        with the checkpoint (`args['soft_nms']`, read by `qpredict.load_model`; the Matrix-NMS form is a list that starts with the string 'mnms').\"\"\"\n"
     "        super().__init__()\n"
     "        self.mnms = None\n"
     "        if soft_nms and soft_nms[0] == \"mnms\":\n"
     "            assert len(soft_nms) == 3, soft_nms\n"
     "            self.mnms = (float(soft_nms[1]), float(soft_nms[2]))                                  # (radius m, sigma): the asserts of `qmnms.matrix_nms_logit` apply when it runs, here at build time\n"
     "            assert 0.0 < self.mnms[0] <= 2.0 and self.mnms[1] > 0.0, soft_nms\n"
     "            soft_nms = None\n"
     "        self.soft_nms = tuple(float(v) for v in soft_nms) if soft_nms else None\n"),
    ("            no[\"e\"] = QN.soft_suppress_logit(no[\"e\"], no[\"pos\"], self.soft_nms[0], self.soft_nms[1], int(self.soft_nms[2]))\n        return no\n",
     "            no[\"e\"] = QN.soft_suppress_logit(no[\"e\"], no[\"pos\"], self.soft_nms[0], self.soft_nms[1], int(self.soft_nms[2]))\n"
     "        elif self.mnms:                                                                           # E292: the same place, the Matrix-NMS decay (e_raw = the head's own logit)\n"
     "            no = dict(no)\n"
     "            no[\"e_raw\"] = no[\"e\"]\n"
     "            no[\"e\"] = QMN.matrix_nms_logit(no[\"e\"], no[\"pos\"], self.mnms[0], self.mnms[1])\n"
     "        return no\n"),
    ("    if phase in (3, 5, 7):\n        l, i = node_loss", "    if phase in (3, 5, 7, 9):\n        l, i = node_loss"),
    ("    if phase in (4, 7):\n        l, i = link_loss", "    if phase in (4, 7, 9):\n        l, i = link_loss"),
    ("    if phase in (5, 7):\n        ce, pl = SE.section_loss", "    if phase in (5, 7, 9):\n        ce, pl = SE.section_loss"),
    ("    if phase in (6, 7):\n        l, i = assign_loss", "    if phase in (6, 7, 9):\n        l, i = assign_loss"),
    ("    for phase in (2, 3, 4, 5, 6, 7):\n        m.zero_grad()\n        loss, info = qgraph_loss(m, b, phase)\n        assert torch.isfinite(loss), (phase, info)\n",
     "    for phase in (2, 3, 4, 5, 6, 7, 9):\n        m.zero_grad()\n        loss, info = qgraph_loss(m, b, phase)\n        assert torch.isfinite(loss), (phase, info)\n"),
    ("    print(\"E277 soft close-node step ok: every phase finite, the gradient reaches the trainable groups\")\n",
     "    print(\"E277 soft close-node step ok: every phase finite, the gradient reaches the trainable groups\")\n"
     "    # E292: the Matrix-NMS form: the same switch, no new parameters, every phase finite, the gradient reaches the trainable groups, none the frozen ones (phase 9 included)\n"
     "    mm = QGraph(width=16, f05=16, d=32, soft_nms=[\"mnms\", 1.75, 0.1])\n"
     "    assert mm.mnms == (1.75, 0.1) and mm.soft_nms is None and set(mm.state_dict()) == set(m.state_dict())\n"
     "    mm.load_state_dict(m.state_dict())\n"
     "    for phase in (3, 4, 5, 6, 7, 9):\n"
     "        mm.zero_grad()\n"
     "        loss, info = qgraph_loss(mm, b, phase)\n"
     "        assert torch.isfinite(loss), (\"mnms\", phase, info)\n"
     "        loss.backward()\n"
     "        assert sum(float(p.grad.abs().sum()) for p in trainable_params(mm, phase) if p.grad is not None) > 0, f\"mnms, phase {phase}: no gradient reaches the trainable parameters\"\n"
     "        frozen_got = [n for n in PHASE_FROZEN[phase] if any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in {\"pyr\": mm.pyr, \"nodes\": mm.nodes, \"links\": mm.links}[n].parameters())]\n"
     "        assert not frozen_got, f\"mnms, phase {phase}: frozen stage {frozen_got} got a gradient\"\n"
     "    assert {n for n in PHASE_TRAIN[9]} == {\"nodes\", \"links\", \"menc\", \"asg\"} and PHASE_FROZEN[9] == (\"pyr\",)\n"
     "    print(\"E292 Matrix-NMS layer ok: every phase finite incl. phase 9, the gradient reaches the trainable groups, none the pyramid\")\n"),
])

# ---- qgates.py (phase 9: numbers only, like phase 7 without the pixel stage)
sub("training/train/qgates.py", [
    ("    if phase == 7:\n        res = {}\n        for k in (2, 3, 4, 5, 6):\n",
     "    if phase in (7, 9):\n        res = {}\n        for k in ((2, 3, 4, 5, 6) if phase == 7 else (3, 4, 5, 6)):                               # E292: phase 9 has a frozen pyramid, so no pixel reading\n"),
    ("        return dict(res, **{\"pass\": True, \"why\": \"phase 7: numbers only\"})", "        return dict(res, **{\"pass\": True, \"why\": f\"phase {phase}: numbers only\"})"),
])

# ---- qgraph_train.py
sub("training/train/qgraph_train.py", [
    ("STEPS = {2: 3000, 3: 6000, 4: 5000, 5: 2000, 6: 1500, 7: 2000, 8: 3000}\nLR = {2: 1e-3, 3: 1e-3, 4: 1e-3, 5: 5e-4, 6: 1e-3, 7: 2e-4, 8: 1e-3}",
     "STEPS = {2: 3000, 3: 6000, 4: 5000, 5: 2000, 6: 1500, 7: 2000, 8: 3000, 9: 1000}\nLR = {2: 1e-3, 3: 1e-3, 4: 1e-3, 5: 5e-4, 6: 1e-3, 7: 2e-4, 8: 1e-3, 9: 2e-4}"),
    ("    ap.add_argument(\"--resume-from\"",
     "    ap.add_argument(\"--soft-nms-mnms\", type=float, nargs=2, default=None, metavar=(\"RADIUS\", \"SIGMA\"), help=\"E292: the one-pass Matrix-NMS layer after the node head (radius m, sigma) in place of the E277 soft step; travels with the checkpoint as args['soft_nms'] = ['mnms', R, sigma]\")\n"
     "    ap.add_argument(\"--resume-from\""),
    ("    a = ap.parse_args()\n    T.seed_all(a.seed)\n",
     "    a = ap.parse_args()\n"
     "    assert not (a.soft_nms and a.soft_nms_mnms), \"--soft-nms and --soft-nms-mnms are two forms of the same switch: give one\"\n"
     "    if a.soft_nms_mnms:\n"
     "        a.soft_nms = [\"mnms\"] + list(a.soft_nms_mnms)                                              # the form that QGraph and qpredict.load_model read\n"
     "    T.seed_all(a.seed)\n"),
])

# ---- run_config.py and startable.py
sub("training/train/run_config.py", [
    ("\"node_loss\", \"zone_bias\", \"soft_nms\")}", "\"node_loss\", \"zone_bias\", \"soft_nms\", \"soft_nms_mnms\")}"),
])
sub("training/startable.py", [
    ("        \"soft_nms\": (list, lambda v: None if len(v) == 3 else \"three values: radius m, smoothness m, passes\", \"E277: the close-node suppression as a soft step\"),\n",
     "        \"soft_nms\": (list, lambda v: None if len(v) == 3 else \"three values: radius m, smoothness m, passes\", \"E277: the close-node suppression as a soft step\"),\n"
     "        \"soft_nms_mnms\": (list, lambda v: None if len(v) == 2 else \"two values: radius m, sigma\", \"E292: the one-pass Matrix-NMS layer after the node head\"),\n"),
])
print("patched copies in", NEW, sorted(os.listdir(NEW)))

# ---- qgraph.py: the synthetic batch of the selftest becomes a function (`_toy_batch`), so that the test of the Matrix-NMS switch (test_mnms_layer.py) uses the same batch; the RNG order of the selftest is unchanged
p = os.path.join(NEW, "qgraph.py")
s = open(p, encoding="utf-8").read()
i0 = s.index("    # a synthetic batch with the real shapes\n")
end_line = '    b["lk_scale"] = torch.full((B, b["lk"].shape[1], 2), float("nan"))\n'
i1 = s.index(end_line, i0) + len(end_line)
body = s[i0:i1]
assert body.count("def ") == 0 and "m = QGraph" not in body
func = "def _toy_batch(B=2):\n    \"\"\"a synthetic batch with the real shapes (two lane pieces A -> B along x = 0, two window events); the batch of the selftest and of `test_mnms_layer.py`\"\"\"\n" + body + "    return b\n\n\n"
s = s[:i0] + "    b = _toy_batch(B)\n" + s[i1:]
j = s.index("def _selftest():")
s = s[:j] + func + s[j:]
open(p, "w", encoding="utf-8", newline="\n").write(s)
print("toy batch extracted:", len(body.splitlines()), "lines")
