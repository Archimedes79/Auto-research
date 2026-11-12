"""one-off: E273 link head, attempt 6: the successors of a node are a distribution over its candidates plus 'none' (row softmax, multi-positive NLL), so that a lane chain does not break
wherever one independent sigmoid falls below 0.5 (a 30 m path has ~15 links: with 0.87 per link only 12 % of the paths are whole)"""


def sub(path, old, new, count=1):
    s = open(path, encoding="utf-8").read()
    assert old in s, (path, old[:80])
    s = s.replace(old, new, count)
    open(path, "w", encoding="utf-8").write(s)


p = "training/grid/qlinks.py"
sub(p, "        self.scale = nn.Sequential(nn.Linear(nin, 64), nn.GELU(), nn.Linear(64, 2))", "        self.scale = nn.Sequential(nn.Linear(nin, 64), nn.GELU(), nn.Linear(64, 2))\n        self.none = nn.Linear(d, 1)                                                               # the logit of 'no successor' of a node (a lane end, a window exit, a false node)")
sub(p, "            self.mlp[-1].bias.fill_(-4.0)", "            self.mlp[-1].bias.fill_(-2.0)\n            self.none.bias.fill_(1.0)")
old = s = open(p, encoding="utf-8").read()
a = s.index("        logit = self.mlp(torch.cat([x0, sa[:, None], sb[:, None], evf], -1))[:, 0]")
b = s.index("def _selftest():")
new = '''        score = self.mlp(torch.cat([x0, sa[:, None], sb[:, None], evf], -1))[:, 0]
        # the successors of a node: a distribution over its candidates and 'none' (row softmax); the link probability is the row probability
        Ld = torch.full((B, K, K), float("-inf"), device=dev, dtype=score.dtype).index_put((pb, pa, pc), score)
        Pr = torch.softmax(torch.cat([Ld, self.none(tok).to(score.dtype)], -1), -1)                  # (B, K, K + 1)
        p_pair = Pr[pb, pa, pc].float().clamp(1e-6, 1 - 1e-6)
        logit = torch.log(p_pair) - torch.log1p(-p_pair)                                          # sigmoid(logit) = the row probability
        out = dict(act=idx, valid=valid, pos=pos, dirv=dirv, tok=tok, cell=torch.stack([ci, cj], -1), pb=pb, pa=pa, pc=pc, is_long=is_long, logit=logit, sa=sa, sb=sb, evidence=evf, pnone=Pr[..., K])
        if gt is not None:
            key = (pb * N * N + fa[pb, pa]) * (N * N) + fa[pb, pc]
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
            out["label"] = lab.float()
            Y = torch.zeros(B, K, K, dtype=torch.bool, device=dev).index_put((pb, pa, pc), lab)
            has = Y.any(-1)
            tgt = torch.where(has, (Pr[..., :K] * Y).sum(-1), Pr[..., K])                       # a node with true successors puts its mass on them, any other node on 'none'
            nll = -torch.log(tgt.clamp_min(1e-6))
            out["row_nll"] = (nll * valid).sum() / valid.sum().clamp_min(1)
            out["gt_missing"] = int(okl.sum()) - int(torch.isin(pair_key, key).sum())          # GT links that did not make it into the candidate list (trimmed or outside K)
        return out


'''
s = s[:a] + new + s[b:]
open(p, "w", encoding="utf-8").write(s)
sub(p, "Link logit = MLP([token i, token j,", "The successors of a node are a DISTRIBUTION over its candidates and 'none' (row softmax; trained with the multi-positive NLL of the true successors, `row_nll`; the link probability is the row probability, `sigmoid(logit)`):\nscore of a pair = MLP([token i, token j,")

# loss
sub("training/grid/qgraph.py", "    return l_b + 2.0 * l_long + 2.0 * l_sc, info", "    if \"row_nll\" in L:\n        info[\"row\"] = float(L[\"row_nll\"])\n        return L[\"row_nll\"] + 2.0 * l_sc, info                                                    # the successor distribution of every node (the BCE terms above are logged only)\n    return l_b + 2.0 * l_long + 2.0 * l_sc, info")
# the read-out of the row probability: successors with a row probability >= 0.25 (a split has two of about 0.4)
sub("training/grid/qpredict.py", "                     a=L[\"sa\"][sel].float().cpu().numpy(), b=L[\"sb\"][sel].float().cpu().numpy(), curves=None)",
    "                     a=L[\"sa\"][sel].float().cpu().numpy(), b=L[\"sb\"][sel].float().cpu().numpy(), curves=None)\n        links[\"p\"] = np.clip(links[\"p\"] / (2.0 * LINK_READ), 0.0, 1.0)                    # the successor distribution is read at LINK_READ: p >= 0.5 <=> row probability >= LINK_READ")
sub("training/grid/qpredict.py", "import qgraph as QG  # noqa: E402", "import qgraph as QG  # noqa: E402\n\nLINK_READ = 0.25                      # output step: a successor counts when its row probability is at least this (the row sums to 1 with 'none': a split has two of about 0.4)")
sub("training/grid/qpredict.py", "(link probability above)", "(link probability above)") if False else None
print("ok")
