"""E292 code step (scratch, scripted text surgery): phase 9 plumbing (fine-tune without the pixel stage, pyramid frozen), the Matrix-NMS form of the soft close-node step (mnms=(R, sigma)), the gate entry, the CLI flag, load_model. Every edit asserts that its anchor exists exactly once."""
import re
def sub(path, pairs):
    s = open(path, encoding="utf-8").read()
    for a, b in pairs:
        assert s.count(a) == 1, (path, a[:70], s.count(a))
        s = s.replace(a, b)
    open(path + ".new", "w", encoding="utf-8").write(s)
    return path

# ---- qgraph.py
sub("training/grid/qgraph.py", [
 ("import qnodes as QN\n", "import qmnms as QMN\nimport qnodes as QN\n"),
 ('PHASE_UPTO = {2: "pyr", 3: "nodes", 4: "links", 5: "sections", 6: "assign", 7: "assign"}', 'PHASE_UPTO = {2: "pyr", 3: "nodes", 4: "links", 5: "sections", 6: "assign", 7: "assign", 9: "assign"}'),
 ("PHASE_TRAIN[8] = (\"ends\",)\n", "PHASE_TRAIN[8] = (\"ends\",)\n"),   # anchor check only
 ("def __init__(self, width=48, f05=32, d=64, soft_nms=None, ends=False):", "def __init__(self, width=48, f05=32, d=64, soft_nms=None, ends=False, mnms=None):"),
 ("        self.pyr = PyramidHead(width=width, f05=f05)\n", "        self.mnms = tuple(float(v) for v in mnms) if mnms else None                                 # E292: the one-pass Matrix-NMS layer (radius m, sigma) after the node head instead of the E277 soft step; a property of the model like soft_nms\n        assert self.mnms is None or (len(self.mnms) == 2 and 0.0 < self.mnms[0] <= 2.0 and self.mnms[1] > 0.0 and self.soft_nms is None), (mnms, soft_nms)\n        self.pyr = PyramidHead(width=width, f05=f05)\n"),
 ("        return no\n\n    def forward(self, batch, upto=\"assign\"", "        if self.mnms:\n            no = dict(no)\n            no[\"e_raw\"] = no[\"e\"]\n            no[\"e\"] = QMN.matrix_nms_logit(no[\"e\"], no[\"pos\"], self.mnms[0], self.mnms[1])\n        return no\n\n    def forward(self, batch, upto=\"assign\""),
 ("    if phase in (2, 7):\n        l, i = pixel_loss", "    if phase in (2, 7):\n        l, i = pixel_loss"),
 ("    if phase in (3, 5, 7):\n        l, i = node_loss", "    if phase in (3, 5, 7, 9):\n        l, i = node_loss"),
 ("    if phase in (4, 7):\n        l, i = link_loss", "    if phase in (4, 7, 9):\n        l, i = link_loss"),
 ("    if phase in (5, 7):\n        ce, pl = SE.section_loss", "    if phase in (5, 7, 9):\n        ce, pl = SE.section_loss"),
 ("    if phase in (6, 7):\n        l, i = assign_loss", "    if phase in (6, 7, 9):\n        l, i = assign_loss"),
])
