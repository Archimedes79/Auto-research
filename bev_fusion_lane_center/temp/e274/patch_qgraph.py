"""one-off (E274): switches of the loss for the recall and long-link arms in training/grid/qgraph.py (defaults = E273, so the E273 numbers reproduce)"""


def sub(path, old, new, count=1):
    s = open(path, encoding="utf-8").read()
    assert old in s, (path, old[:90])
    open(path, "w", encoding="utf-8").write(s.replace(old, new, count))


p = "training/grid/qgraph.py"
sub(p, "W = dict(pix=1.0, node=1.0, link=1.0, sec=1.0, asg=1.0)",
    "W = dict(pix=1.0, node=1.0, link=1.0, sec=1.0, asg=1.0)\n"
    "# E274 switches (the defaults are the E273 loss): ev_w = weight of the 1 m event heat-map loss in the pixel stage; node_w_evt / node_w_inter = extra weight of the POSITIVES of the existence loss\n"
    "# at event cells / at cells inside the (grown) intersection mask (weight 1 + w_evt * event + w_inter * inter); long_match = weight of the dual-softmax loss over the long-link candidates (E274)\n"
    "CFG = dict(ev_w=1.0, node_w_evt=0.0, node_w_inter=0.0, long_match=0.0)")
sub(p, "def focal_bce(logit, y, alpha=0.5, gamma=2.0, norm=None):\n    p = logit.sigmoid()\n    ce = F.binary_cross_entropy_with_logits(logit, y, reduction=\"none\")\n    pt = p * y + (1 - p) * (1 - y)\n    l = ce * (1 - pt) ** gamma * (alpha * y + (1 - alpha) * (1 - y))",
    "def focal_bce(logit, y, alpha=0.5, gamma=2.0, norm=None, pos_w=None):\n    p = logit.sigmoid()\n    ce = F.binary_cross_entropy_with_logits(logit, y, reduction=\"none\")\n    pt = p * y + (1 - p) * (1 - y)\n    a_pos = alpha if pos_w is None else alpha * pos_w                                   # pos_w: a per-element weight of the positives (E274)\n    l = ce * (1 - pt) ** gamma * (a_pos * y + (1 - alpha) * (1 - y))")
sub(p, "    return l1 + l_ev + l_ln + l_rd, info", "    return l1 + CFG[\"ev_w\"] * l_ev + l_ln + l_rd, info")
sub(p, "    l_e = focal_bce(no[\"e\"].float(), ex, norm=npos)",
    "    wmap = None\n    if CFG[\"node_w_evt\"] or CFG[\"node_w_inter\"]:                                           # E274: recall where the nodes are weak (event cells, intersections)\n"
    "        evt = (batch[\"flags\"].to(dev).float().max(1)[0] > 0.5).float()\n"
    "        inter = (batch[\"road2\"].to(dev).float()[:, 1] > 0.5).float()\n"
    "        wmap = 1.0 + CFG[\"node_w_evt\"] * evt + CFG[\"node_w_inter\"] * inter\n"
    "    l_e = focal_bce(no[\"e\"].float(), ex, norm=npos, pos_w=wmap)")
sub(p, "    return l_b + 2.0 * l_long + 2.0 * l_sc, info                                                            # the degree term is logged, not trained: with one-to-many labels a node has several true successors",
    "    l_m = torch.zeros((), device=dev)\n"
    "    if CFG[\"long_match\"] > 0 and lg.any():                                                       # E274: the long-link candidates of one entry (or one exit) compete, with a 'none' entry\n"
    "        l_m = long_match_loss(L)\n"
    "        info[\"match\"] = float(l_m)\n"
    "    return l_b + 2.0 * l_long + 2.0 * l_sc + CFG[\"long_match\"] * l_m, info                                  # the degree term is logged, not trained: with one-to-many labels a node has several true successors")
# the dual-softmax loss, before link_loss
sub(p, "def link_loss(L, batch):", '''def _seg_lse(z, gid, G, extra):
    """log(sum_j exp(z_j) + exp(extra)) per group (a segment log-sum-exp with one fixed 'none' logit per group)"""
    m = torch.full((G,), float("-inf"), device=z.device).scatter_reduce(0, gid, z, "amax", include_self=True)
    m = torch.maximum(m, torch.full_like(m, extra))
    s = torch.zeros(G, device=z.device).scatter_add(0, gid, torch.exp(z - m[gid])) + math.exp(extra) * torch.exp(-m)
    return m + torch.log(s)


def long_match_loss(L, none_logit=0.0):
    """Dual softmax over the long-link candidates (SuperGlue / LoFTR style, one 'none' entry per row and per column, no iterations): the candidates of one source node compete, so do the
    candidates of one destination node. Multi-positive NLL (a node may have several true successors: the mass of the true ones adds up); a node without a true long link goes to 'none'.
    Training signal only: the read-out still thresholds the independent link probability."""
    lg = L["is_long"]
    z = L["logit"].float()[lg]
    y = (L["label"][lg] > 0.5).float()
    K = L["valid"].shape[1]
    total = z.new_zeros(())
    for gid_ in (L["pb"][lg] * K + L["pa"][lg], L["pb"][lg] * K + L["pc"][lg]):                 # rows: the source node, columns: the destination node
        _, inv = torch.unique(gid_, return_inverse=True)
        G = int(inv.max()) + 1
        lse = _seg_lse(z, inv, G, none_logit)
        pos_mass = torch.zeros(G, device=z.device).scatter_add(0, inv, torch.exp(z - lse[inv]) * y)
        has = torch.zeros(G, device=z.device).scatter_add(0, inv, y) > 0
        nll = torch.where(has, -torch.log(pos_mass.clamp_min(1e-6)), lse - none_logit)           # -log(P(none)) = lse - none_logit
        total = total + nll.mean()
    return total / 2


def link_loss(L, batch):''')
print("ok")
