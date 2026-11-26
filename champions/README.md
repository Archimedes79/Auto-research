# champions - the facts of each problem's best network and its comparison networks

Created on 2026-11-26 by step S0 of the restructure plan (`../autoresearch/INTEGRATION_2026-11-26.md`); filled per problem in steps S4 (BEV) and S5 (lane). Protected: written by `research adopt` on the owner's word.

**No code here.** A champion's code is in its problem folder on `main` (`../problems/<name>/`), exactly in its tag, and the anchor test there reproduces the numbers below from the weights. Its weights and predictions are on D: (`D:\Auto-research\runs\<name>\`), named here with their hashes.

```
<name>/
  champions.json      who is champion, who is a comparison network, and the decision for each
  <network>/
    network.json      tag and commit, config, weights with sha256, the command; numbers per metric with split and denominator;
                      evaluator hash, ground-truth signature; timing (ms per frame, parameters, machine)
    counts.json       per-scene counts: paired comparisons and report tables without running the network
    architecture.svg  the drawing (text, made by archdraw)
```
A report takes numbers, counts, timing and drawing from here and draws its pictures again from the stored predictions and the raw data (`../autoresearch/report/` and the problem's `evaluation/views.py`). A replaced champion stays as a comparison network; it is evaluated again from a worktree of its tag.

| folder | champion | state |
|---|---|---|
| `lane_center/` | E372 S4 (D120) | facts come in step S5; today described in `../bev_fusion_lane_center/winner/` |
| `bev_fusion/` | `t21_ctl` or `t20_base_s0`: open | facts come in step S4; today described in `../bev_fusion/winner/` |
