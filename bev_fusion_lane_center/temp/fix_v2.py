"""one-off: E273 head revision after the failed gate of phase 4 (attempt 1): the node head reads the predicted pixel maps, stronger flag / long-link-endpoint losses, one-to-many link labels
(a node counts for the GT node nearest to it, within 1.5 m), more candidates for long links, more steps"""


def sub(path, old, new, count=1):
    s = open(path, encoding="utf-8").read()
    assert old in s, (path, old[:80])
    s = s.replace(old, new, count)
    open(path, "w", encoding="utf-8").write(s)


# --- node head: reads the predicted centre / event probability maps (pooled to 2 m)
sub("training/grid/qnodes.py", "        self.inp = nn.Sequential(nn.Conv2d(c2 + c1 + c05 + 2 + 16, d, 1), nn.GroupNorm(8, d), nn.GELU())",
    "        self.inp = nn.Sequential(nn.Conv2d(c2 + c1 + c05 + 2 + 16 + 8, d, 1), nn.GroupNorm(8, d), nn.GELU())")
sub("training/grid/qnodes.py", "    def forward(self, feat2, feat1, feat05, road2):\n        B = feat2.shape[0]\n        c = cell_centres(feat2.device).expand(B, -1, -1, -1)\n        t = self.inp(torch.cat([feat2, F.avg_pool2d(feat1, 2), F.avg_pool2d(feat05, 4), road2.sigmoid(), fourier(c[:1]).expand(B, -1, -1, -1)], 1))",
    "    def forward(self, feat2, feat1, feat05, road2, centre1):\n        B = feat2.shape[0]\n        c = cell_centres(feat2.device).expand(B, -1, -1, -1)\n        cen, ev = centre1[:, :1].float().sigmoid(), centre1[:, 6:12].float().sigmoid()           # the pixel stage's own centre-lane and event probabilities, pooled to the 2 m grid\n        pm = torch.cat([F.avg_pool2d(cen, 2), F.max_pool2d(cen, 2), F.avg_pool2d(ev, 2)], 1)\n        t = self.inp(torch.cat([feat2, F.avg_pool2d(feat1, 2), F.avg_pool2d(feat05, 4), road2.sigmoid(), fourier(c[:1]).expand(B, -1, -1, -1), pm.to(feat2.dtype)], 1))")
sub("training/grid/qnodes.py", "    NodeHead()(feat2, feat1, feat05, road2) -> dict", "    NodeHead()(feat2, feat1, feat05, road2, centre1) -> dict")
sub("training/grid/qnodes.py", "The token of a cell = the 2 m feature + the pooled 1 m / 0.5 m features + road and intersection probabilities + a Fourier code of the cell centre;", "The token of a cell = the 2 m feature + the pooled 1 m / 0.5 m features + road and intersection probabilities + the pooled centre-lane and event probabilities of the pixel stage (8 channels) + a Fourier code of the cell centre;")
sub("training/grid/qnodes.py", "    road = torch.randn(B, 2, 50, 50)\n    m = NodeHead(c2=24, c1=16, c05=8, d=32, heads=4, layers=2, k=7)\n    o = m(f2, f1, f05, road)", "    road = torch.randn(B, 2, 50, 50)\n    cen1 = torch.randn(B, 12, 100, 100)\n    m = NodeHead(c2=24, c1=16, c05=8, d=32, heads=4, layers=2, k=7)\n    o = m(f2, f1, f05, road, cen1)")
sub("training/grid/qgraph.py", 'no = run("nodes", self.nodes, P["feat2"], P["feat1"], P["feat05"], P["road2"])', 'no = run("nodes", self.nodes, P["feat2"], P["feat1"], P["feat05"], P["road2"], P["centre1"])')
sub("training/train/qgates.py", "    no = model.nodes(P[\"feat2\"], P[\"feat1\"], P[\"feat05\"], P[\"road2\"])", "    no = model.nodes(P[\"feat2\"], P[\"feat1\"], P[\"feat05\"], P[\"road2\"], P[\"centre1\"])")

# --- node loss: every flag counts fully, long-link endpoints weigh more
sub("training/grid/qgraph.py", "    l_f = (F.binary_cross_entropy_with_logits(no[\"flags\"].float(), fl, reduction=\"none\", pos_weight=torch.tensor(3.0, device=dev)) * m[:, None]).sum() / (npos * 8)",
    "    l_f = (F.binary_cross_entropy_with_logits(no[\"flags\"].float(), fl, reduction=\"none\", pos_weight=torch.tensor(6.0, device=dev)) * m[:, None]).sum() / npos")
sub("training/grid/qgraph.py", "    return l_e + l_f + l_o + 0.5 * l_d + 0.5 * l_s + 0.5 * l_u, info", "    return l_e + 0.5 * l_f + l_o + 0.5 * l_d + 0.5 * l_s + 2.0 * l_u, info")
# --- link loss: no degree term (one-to-many labels make the GT degree meaningless), scales stay
sub("training/grid/qgraph.py", "    return l_b + 0.5 * l_deg + 2.0 * l_sc, info", "    return l_b + 2.0 * l_sc, info                                                            # the degree term is logged, not trained: with one-to-many labels a node has several true successors")

# --- link head: one-to-many labels, more long-link endpoints
sub("training/grid/qlinks.py", "    def forward(self, no, cen, gt=None, K=512, KL=32, thr=0.05):", "    def forward(self, no, cen, gt=None, K=512, KL=64, thr=0.05):")
sub("training/grid/qlinks.py", "            key = (pb * N * N + fa[pb, pa]) * (N * N) + fa[pb, pc]\n            out[\"label\"] = torch.isin(key, pair_key).float()",
    """            key = (pb * N * N + fa[pb, pa]) * (N * N) + fa[pb, pc]
            lab = torch.isin(key, pair_key)
            if "offset" in gt:                                                                  # one-to-many: a node counts for the GT node nearest to it (within 1.5 m)
                gx = gt["exists"].flatten(1) > 0.5
                gp = cell_centres(dev).flatten(2).transpose(1, 2) + gt["offset"].flatten(2).transpose(1, 2)           # (B, N*N, 2) GT node positions
                dist_g = torch.cdist(pos.float(), gp.float()).masked_fill(~gx[:, None, :], float("inf"))
                dmin, gidx = dist_g.min(-1)
                gmap = torch.where(dmin <= 1.5, gidx, torch.full_like(gidx, -1))                 # (B, K) flat cell of the nearest GT node
                ga_, gc_ = gmap[pb, pa], gmap[pb, pc]
                kgeo = (pb * N * N + ga_.clamp_min(0)) * (N * N) + gc_.clamp_min(0)
                lab = lab | (torch.isin(kgeo, pair_key) & (ga_ >= 0) & (gc_ >= 0) & (ga_ != gc_))
            out["label"] = lab.float()""")
sub("training/grid/qlinks.py", "from qnodes import N, sample, unit", "from qnodes import N, cell_centres, sample, unit")
sub("training/grid/qgraph.py", "gtd = {k: batch[k].to(dev) for k in (\"exists\", \"uout\", \"uin\", \"lk\")} if gt else None", "gtd = {k: batch[k].to(dev) for k in (\"exists\", \"uout\", \"uin\", \"lk\", \"offset\")} if gt else None")
# --- steps
sub("training/train/qgraph_train.py", "STEPS = {2: 3000, 3: 3000, 4: 3000, 5: 1500, 6: 1500, 7: 2000}", "STEPS = {2: 3000, 3: 6000, 4: 5000, 5: 2000, 6: 1500, 7: 2000}")
print("ok")
