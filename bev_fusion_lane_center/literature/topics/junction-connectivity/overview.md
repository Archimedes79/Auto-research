# Overview - junction-connectivity: which entry lane joins which exit lane (2026-11-10)

Built from the notes listed below only; regenerate rather than edit by hand. This page is the input of experiment planning: read the digest and "For experiment planning", follow a link only when you need the detail. Links are relative to this file. Reference values of our head (junction F1: oracle pairs 0.88, all pairs 0.53, learned pairs 0.71) are the brief's, quoted in the port cards of gumbel-sinkhorn and sept25; learned to oracle is a gap of 0.17 [derived]. K196-K202 are proposed ids (the register ends at K195, not edited here). [derived] / [estimate] / [judgement] are the notes' own tags.

**Digest** (the three impulses a planner should take, one line each, with why now):
1. Test an end-to-end entry-to-exit assignment first: log-domain Sinkhorn with a dustbin (K196, gumbel-sinkhorn) against independent pair sigmoids, after the free degree count of the ground-truth pairs; why now: pairs are the junction gap (0.71 vs oracle 0.88), the block is about 15 lines and one scalar, and the paper's one-to-one setting may not hold at forks.
2. Feed the E273 successor edges into the pair logits as a log-prior (K198, our idea written in the sept25 port card, not in any paper, untested): the only candidate here that uses what we already have, since SEPT never uses SD links between roads (sept25); why now: the tokens carry the edges, the toy case goes F1 0.667 -> 1.0 (toy only), the real size depends on registration trust (K189).
3. Expect no ready pair mechanism: audit our pair loss against the generic recipe (K197: positives x3, endpoint coupling, matched-only supervision; can22-minimalcycles) in about a day, and leave the cycle target (K200: C-F +0.1 on nuScenes) and the keypoint auxiliary (K199: +1.2 of +6.6 TOP_lsls) until pairs stay below the oracle; why now: it is cheap and tells whether the 0.71 already contains these ingredients.

## State of the art (best methods, best numbers, on which data)
| method | idea | best number | data / metric | code | note |
|---|---|---|---|---|---|
| SEPT (on TopoLogic / TopoNet) | SD-map raster + vector fused into the BEV feature, auxiliary junction heatmap; pair heads unchanged | TOP_ll (v1.1) 31.2 / 19.5, without SEPT 23.6 / 10.9 | OpenLane-V2 subset_A val, ResNet-50, one run each, no intervals | none released | sept25 |
| Minimal-cycle target + pair MLP (transformer) | Bezier queries, pair logit per ordered query pair, cycle cover as aux target | C-F 55.3 nuScenes (55.2 without), 58.1 Argoverse (54.9 without) | front camera; C-F is global over curves, not per junction | partial, no licence | can22-minimalcycles |
| Gumbel-Sinkhorn | Sinkhorn operator makes a permutation differentiable; Hungarian at test time | sorting: proportion of sequences with an error 0 at N = 5, 10, 15, 80, 100 (Table 1); jigsaw 5x5 Kendall tau .78 (Table 2) | synthetic or small, square; no lane data | Apache-2.0, TF1, archived | gumbel-sinkhorn |

The rows use different data and metrics and are not comparable; no row has a junction-only pair metric. In the corpus and not re-read here: TopoLogic (K30, TP01) and UniTopo (K34, TP06; TOP_ll 30.1 / 31.8 on the two OpenLane-V2 subsets per candidates.md, abstract page, metric version not stated there).

## Techniques ranked by expected transfer to our product
| K id | technique | where it would plug in | expected effect | cost | evidence |
|---|---|---|---|---|---|
| K196 (new) | log-domain Sinkhorn with a learned dustbin on the entry x exit score matrix, NLL on log P; capacities where degree > 1 | pair logits of the junction head; soft P weights the candidate edges, hard step only on the output (D55) | unknown: the paper has no control against independent scores | S (dustbin), M (capacities); ~40 small kernels at 20 iterations [estimate] | gumbel-sinkhorn; source Mena et al. 2018 + SuperGlue dustbin (C30); status checked; neighbours K145, K189 |
| K198 (new) | map adjacency C as log-prior: logit_ij += log(eps + C[piece(i), piece(j)]) | pair logits; piece(.) from registration and trust (K188, K189); E273 tokens (K17, K73) | toy F1 0.667 -> 1.0, falls with a wrong map; real size unknown | not estimated in the note; inputs exist | sept25 (port card); status own idea, derived, untested |
| K197 (new) | pair-head recipe: ordered concat of node embeddings, MLP to one logit, BCE positives x3, endpoint coupling loss, matched-only labels | our pair scorer, as an audit of the loss | unknown; paper has no ablation of these details | 0.5-1 day, compute negligible | can22-minimalcycles; source CVPR 2022 paper and code; status checked; neighbour K30 |
| K199 (new) | junction keypoint heatmap auxiliary (IKPD-like) on the shared grid feature, targets from the map or GT lane graph | side branch of the grid feature | +1.2 of +6.6 TOP_lsls (LaneSegNet host, one run); may duplicate our junction head (K180) | S | sept25; source SEPT 2025; status checked |
| K201 (new) | crossing-pattern consistency: loss on soft pair probabilities times the geometric crossing indicator of the candidate paths | loss on P | none; sketch only | not estimated; card first | can22-minimalcycles (port card); status own idea, untested |
| K200 (new) | minimal-cycle cover as an auxiliary set-valued target | auxiliary head; needs query slots, our head is dense | C-F +0.1 nuScenes, +3.2 Argoverse; no junction-only metric | GT cycles 1-2 d, head 2-3 d [estimate] | can22-minimalcycles; status checked |
| K202 (new) | gated fusion (DGFF) of a raster and a vector map branch | only if both E104b-style and E273 branches feed one feature | TOP_lsls 31.0 vs 29.5 / 29.9 / 30.0 for addition / concat / cross-attention (one run each) | S to M | sept25; status checked; conditional |

## For experiment planning
| candidate | gap of PROBLEM.md it attacks | K id / note | variant or block | cost (GPU h, days) | first test |
|---|---|---|---|---|---|
| 0. degree count of GT pairs | decides the form of the assignment | K196 / gumbel-sinkhorn (port card sec. 0) | CPU script on stored GT: share of junctions with all row and column sums <= 1, in/out-degree histograms, entries without exit | hours, no GPU; risk none | report the shares; the threshold (the note proposes 90 % partial permutations) is fixed by the planner before the run |
| 1. Sinkhorn assignment vs independent pairs | junction pairs 0.71 vs oracle 0.88 | K196 / gumbel-sinkhorn | arms on the same features and labels: A current, B independent sigmoids (control), C row softmax + bin, D Sinkhorn + dustbin L = 20 tau = 1, E L in {1, 3, 10}, F capacities | S for D, M for F; GPU h not estimated; risk: one-to-one mismatch at forks, soft output needs a tuned threshold | 2 seeds, VAL, paired scene bootstrap, band 0.010: junction F1, topo F1, ms per frame, share of junctions where rounding changes the pair set; no expected size from the source |
| 2. map prior on pair logits | same gap, from the map side | K198 / sept25 | prior on vs off; also with a shifted or wrong map | not estimated in the note, inputs exist; risk: wrong or outdated map, depends on K189 trust | junction F1 on VAL with the real registration; source gives a toy case only |
| 3. pair-loss audit | same gap, cheapest | K197 / can22-minimalcycles | compare our loss with weight x3, endpoint coupling, matched-only; ablate each | 0.5-1 day; risk low | junction F1; no expected size from the source |
| 4. junction keypoint auxiliary | node and junction support for pairs | K199 / sept25 | targets from the GT lane graph, not SD-map points | S; risk: duplicates existing node supervision | junction F1 and node recall vs the current head; source +1.2 TOP_lsls on one host |
| 5. crossing-pattern consistency | global consistency of pairs | K201 / can22-minimalcycles | loss on P x crossing indicator | not estimated; risk: untested | write a card first |
| 6. cycle cover target | global consistency of pairs | K200 / can22-minimalcycles | only if pairs stay below the oracle after rows 1-3 | GT cycles 1-2 d + head 2-3 d [estimate]; risk high (C-F +0.1 on nuScenes) | none before rows 1-3 |

## Detail links
The generated table of all notes (title, summary, relevance, status, paper, code): [table.md](table.md). Check of the notes by a fresh reader: [check.md](check.md) (46 claims confirmed, none contradicted or unreadable). Scan and leftovers: [candidates.md](candidates.md), [deferred.md](deferred.md) (autograph23, survey-yao25, roadtopoformer23). The notes this overview rests on, one line each:
- [gumbel-sinkhorn Learning Latent Permutations with Gumbel-Sinkhorn Networks](gumbel-sinkhorn/note.md) (port card: [port_card.md](gumbel-sinkhorn/port_card.md)) - K196, the assignment block and its first experiment
- [sept25 SEPT: SD-map prior and intersection keypoints](sept25/note.md) (port card: [port_card.md](sept25/port_card.md)) - K198 idea, K199, K202; shows the map helps topology only through the feature
- [can22-minimalcycles Minimal-cycle topology supervision](can22-minimalcycles/note.md) (port card: [port_card.md](can22-minimalcycles/port_card.md)) - K197, K200, K201; the plain pair head
- Register of K ids: [register.md](../../register.md)

## Open questions the literature does not answer
- **No pair mechanism beyond the generic pair head.** None of the three notes gives a dedicated entry-to-exit scorer: can22-minimalcycles is an MLP on query pairs with BCE; sept25 leaves the host pair heads unchanged; Sinkhorn is a relaxation, not a score. Dedicated scorers already in the corpus (K30 TopoLogic, K34 UniTopo) were not re-read.
- **SEPT only improves the BEV feature.** No SD link-to-link connection is used, and there is no junction or non-junction split of TOP_ll (sept25).
- **The cycle loss has no junction-only metric.** C-F is global, MC-F and order are metrics the loss trains for, single runs (can22-minimalcycles).
- **Sinkhorn is square and one-to-one** unless a dustbin or capacities are added; its paper has no control against independent pair scores, no lane data, no runtime (gumbel-sinkhorn).
- Also open: how entries and exits of a junction are found, calibration of P, consistency across junctions, speed on our GPU, and whether competition beats independent scores at all (gumbel-sinkhorn).

## What not to do (tried by us or shown not to work, with ids)
- Do not train with a small temperature: tau = 0.01 gives Kendall tau .62 and tau = 100 gives .47 against .78 in Table 2 (gumbel-sinkhorn, Table 4, CelebA 5x5); keep tau = 1 and let the scores scale (K196).
- Do not put Hungarian or any hard matching on the training path; hard rounding only on the output (D55). The Gumbel noise and the latent reconstruction loss are not needed with labelled pairs (gumbel-sinkhorn).
- Do not start with the cycle head (K200): C-F +0.1 on nuScenes and no isolation of the cause (can22-minimalcycles).
- Do not take SEPT as a pair fix: its gain sits in the feature; IKPD is ablated on one host only (sept25).
- Own earlier negatives on the node side, for orientation (register, not these notes): K182 fork-coverage loss and K183 assembly-side fork recovery are registered as negative.
