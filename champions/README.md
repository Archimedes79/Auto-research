# champions - the facts of each problem's best network and its comparison networks

Protected (`research grant` under a decided decision for a hand edit); written by `research adopt` (a compared arm with a KEEP) or
`research register` (a measured network), both on the owner's decided decision.

**No code here.** A champion's code is in its problem folder (`../problems/<name>/`), exactly in its experiment's snapshot tag; `research anchor`
evaluates it again from the stored weights and must give the numbers below. Weights and predictions are in the runs root of the machine file
(on this PC `D:\Auto-research\runs\<name>\`).

```
<name>/
  champions.json      who is champion, who is a comparison network, and the history with the decision of each entry
  <experiment>.<arm>/
    network.json      per seed: the run, its config, the weights (run folder, checkpoint, sha256), split, frames, scenes, the numbers of every
                      metric, the per-scene counts (paired comparisons and report tables without running the network), the evaluation's
                      identity (evaluator hash, ground-truth signature, snapshot commit), the cost the evaluation measured
    architecture.svg  the problem's drawing at adoption, when it has one (training/architecture.svg)
```
A report (`research report`) takes numbers and counts from here and draws its pictures again from the stored predictions and the raw data.
A replaced champion stays as a comparison network.

| folder | champion | state |
|---|---|---|
| `lane_center/` | `E396.champ` = E372 S4 (D120) | registered 2026-11-27: VAL topo F1 0.6682 / 0.6813 (seed 0 / 1), the v1 numbers to the last digit |
| `bev_fusion/` | `t21_ctl` or `t20_base_s0` | open: the owner's choice (`../problems/bev_fusion/DECISIONS.md`) |
