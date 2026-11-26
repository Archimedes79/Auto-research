# DESIGN - lane-centre graph from camera + radar (lcr)

One small file for people and for AI: the modules of the chain at a glance and the rules it obeys. With it a coder can build a similar chain; the exact specification is the code (paths in the table). Every rule can be attacked: a card names the rule and a new mechanism or a cheaper test, and the written verdict changes the rule here. The goal, data, protocol and numbers are in `PROBLEM.md`, the owner's permissions in `experiment_execution/DECISIONS.md`.

## The chain

```
camera + radar (ONE frame, ONE radar scan)
  -> 1 Foundation     frozen GaussianCaR BEV features, cached
  -> 2 Lane head B'   frozen pyramid + node head: a candidate node per 2 m cell
  -> 3 Tokens         the top-512 nodes become tokens
  -> 4 Token head     self-attention among the tokens; the wiring IS the attention; node types
  -> 5 Read-out       four named thresholds
  -> lane graph       nodes, links, types in the ego frame
```
The model lives in `net/` in three layers, apart from the training system: `net/prep/` (targets, datasets, caches) < `net/model/` (the networks) < `net/loss/` (the losses) < `training/` (R13).

| module | job | code |
|---|---|---|
| 1 Foundation | camera + radar -> a BEV feature map (39 channels, 200 x 200, 0.5 m); frozen, cached | GaussianCaR (frozen) |
| 2 Lane head B' | the pyramid and the node head: existence, position, heading and type flags of a node per 2 m cell | `net/model/qgraph.py` (`QGraph`), `net/model/pyramid.py` |
| 3 Tokens | the top-K (512) nodes become tokens (position, direction, existence, node token) | stage 1 cached: `net/prep/lane_tokcache.py`; stage 2 live, node head trainable: `training/train/lane_live_tok.py` |
| 4 Token head | three self-attention layers (relative-geometry bias, allowed-pair gate); per source a softmax over successors plus a `none` slot, for short (<= 8 m) and long links; a seven-way node-type softmax | `net/model/jtoken.py` (`JTokenHead`) |
| 5 Read-out | link short 0.3, link long 0.2, node kept when p_none < 0.6, none gate off -> the graph | `evaluation/jtoken_eval.py` |
| Losses (training) | the token likelihood with the `none` slot, node types and handle scales; B''s own node loss | `net/loss/loss_jtoken.py`, `net/loss/loss_qgraph.py` |

Training: stage 1 trains the head on cached tokens (`training/train/jtoken_train.py`); stage 2 trains the head and the node head of B' together through the token loss (`training/train/jlive_train.py`).
Not in the chain: the SD-map route and a past state (rules R2 and R9). The earlier system E324 s (`training/lanegraph/`) is a reference, never a bar.
Headline, VAL 873 frames: topo F1 0.668 / 0.681 (two seeds), 16.2 ms on an RTX 3060, 2.3 M parameters; current best and the reference rows are in `PROBLEM.md`.

## Rules
| rule | why | open to attack when |
|---|---|---|
| R1 **End-to-end differentiable:** soft steps up to the output, a loss on the output reaches every module, matching inside the loss is allowed, hard selection (top-K, thresholds) only on the output and named. A candidate that adds a rule step is no solution | the owner: D55, D88 | the owner changes it. Today GaussianCaR and the pyramid are frozen and the top-K, the pair gate, the 8 m split and the four thresholds stay named |
| R2 **One frame and one radar scan per step;** the past only as a recursive state at feature level | the owner: D90, D91. Tried: feature state, token cross-attention, gated fusion (E354, E368, E387): no gain, so the champion has no state | the owner, or a new mechanism for the nodes the single frame misses |
| R3 **Simple, cheap, fast:** ms per frame on one RTX 3060 (batch 1, cached features), at most 14.3 vertices per 100 m (not yet checked on that basis for the token champion); inside the card's band the simpler AND faster system wins; the mission is a good, simple, cheap champion, 'okayish' is enough | the owner: D106, D113 | the owner |
| R4 **One token set:** the predicted nodes are the tokens, the wiring is attention, the node type a softmax, a `none` slot per class | D104, D113; E355, E365. Tried: iterative wiring with early reads under a loss and messages along the soft links (E392, r2): +0.007 topo F1, replicated in four head-seed pairs, but inside the band and +2 ms or more for the whole path (the cost bar of +1.0 ms per round missed), so the one-read head stays (D138) | a system inside the band that is simpler and faster, or clearly better at <= 16.2 ms |
| R5 **Softmax wherever a selection can be one;** the hard top-K (512) only as a named compute budget | D113; E383: K 384 is faster but loses node coverage | the pair attention itself (a fused kernel or a sparser pair set that keeps the GT links) |
| R6 **Two-stage training:** the head on cached tokens, then head and B' node head together through the token loss | E372, D120: not worse than B' frozen, no gain shown; kept because of R1 | a card in which B' frozen wins by more than 0.01, or the owner waives that the loss reaches B' |
| R7 **The four decode thresholds are hand-tuned on TRAIN and named;** learning them by a soft-F1 surrogate loses | E388, D115 | a loss that sees the graph metric (delivered lanes, junction precision) |
| R8 **Matching stays inside the loss;** matching-free designs are admissible, one lost so far (old chain) | D55; E244, E255 | a calibration of the validity scores before thinning |
| R9 **Perception first, map later, step by step:** the SD map joins as a logical road graph (lanes per direction, connections), encoded first, fusion second, error models third; the owner sees each result; a wrong map must not be worse than no map | the owner: D130. Tried: one cross-attention to route samples read the map but missed its bar (E389) | the bars of E390 step 1 |
| R10 **Heads keep their width (64 channels);** how small they can be is measured by importance pruning, not by narrowing | the owner: D92, D99 | the size-against-loss table of D99 |
| R11 **Ground truth:** junction entry and exit points as late as possible, but only where the network can see them | the owner: D71; E283: later than the end of the map's lane has no support in the data | a definition of 'can see' |
| R12 **Protocol:** TEST once per finished design with the owner's recorded OK; select and tune on TRAIN / VAL only; the D115 cell (batch 4, 873 VAL frames), noise band 0.010 topo F1, 2 seeds, paired scene bootstrap | hard rule 1; D115; CLAUDE.md "Numbers" | the owner |
| R13 **The model is apart from the training system:** preprocessing (`net/prep`) < network (`net/model`) < losses (`net/loss`) < `training/`; a new import against this order is refused (`net/tests/test_net_layering.py`, no exceptions since 2026-11-26; only a command line may hand a preprocessing step the stored model it runs); runs are started, stopped and watched by the tooling in `../autoresearch/` | the owner: D152 | the owner |

## How this file moves
A card names the rules it attacks or changes (its Design delta); the coder is given this file plus the delta. KEEP: the delta becomes the rule and the card's id the reason (the old text stays in git); CUT or TIE: the rule stays and records the result of the attack. An instruction of the owner that fixes how the system must be becomes a rule, his words or the decision id in the reason; permissions go to `DECISIONS.md`. The evidence stays in the cards and the ledger; the first, evidence-heavy version of this file is at git `6b160e66`.
