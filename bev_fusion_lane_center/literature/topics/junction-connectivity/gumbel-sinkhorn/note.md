# gumbel-sinkhorn Learning Latent Permutations with Gumbel-Sinkhorn Networks

- title: Learning Latent Permutations with Gumbel-Sinkhorn Networks
- relevance: 2
- summary: Mena et al. (ICLR 2018) make a latent permutation learnable end to end: the hard matching is replaced by the Sinkhorn operator S(X/tau) (row/column normalisation, L = 20), Gumbel noise gives samples, and the Hungarian algorithm rounds at test time. For us it is a tiny differentiable entry-to-exit assignment on top of node features, but the paper covers only square one-to-one matchings: unmatched entries or exits need a SuperGlue-style dustbin row and column, and one-to-many junction pairs need capacities. About 15 lines and one extra scalar; first test: count the degrees of the ground-truth pairs, then an arm against independent pair sigmoids.
- status: checked
- paper: https://arxiv.org/abs/1802.08665
- code: https://github.com/google/gumbel_sinkhorn
- verdict: use
- result: -

| | |
|---|---|
| venue / year | ICLR 2018 (arXiv v1 2018-02-23; journal reference on the abstract page) |
| groups | A3: 2, A4: 2, F3: 1 |
| licence | code: Apache-2.0, archived 2022-09-07, TensorFlow 1 (google/gumbel_sinkhorn); PyTorch ports: perrying/gumbel-sinkhorn (licence not shown on the page), ThinkMatch (Mulan PSL v2) |
| why the verdict | The Sinkhorn operator (not the Gumbel / VAE part) is the cheapest differentiable entry-to-exit assignment and a clean arm against our learned pairs; use it only with a dustbin and after the degree count, because the paper's one-to-one constraint may not hold at junctions |

**What it does** (read: abstract page, full text through the page fetch tool; details in [method.md](method.md)): The paper wants to learn a permutation P that is a latent variable (sort numbers, solve jigsaws, identify neurons), where the hard matching M(X) = argmax over permutations of the score is not differentiable. It replaces M(X) by the Sinkhorn operator S(X/tau): exponentiate, then alternate row and column normalisation until the matrix is doubly stochastic; Theorem 1 says S(X/tau) maximises the score plus tau times the entropy over the Birkhoff polytope and tends to M(X) as tau goes to 0. Gumbel noise added before the operator, S((X + eps)/tau), gives reparametrised samples of near-permutations and a tractable KL for a variational use. The score matrix X = g(input, theta) comes from a small network, the loss is a reconstruction loss on the un-permuted data, 20 Sinkhorn iterations are unrolled, and the Hungarian algorithm gives the hard permutation at test time.

**Results that matter** (all from the paper's arXiv HTML text, read through the fetch tool; tables were read on two renderings and agree): sorting, proportion of sequences with any error, U(0,1): 0 at N = 5, 10, 15, 80, 100 and .01 at N = 120, against .06 / .43 / .9 for Vinyals et al. 2015 at N = 5 / 10 / 15 (Table 1). Jigsaw, CelebA, Kendall tau 1.0 / .96 / .88 / .78 for 2x2 to 5x5 (Table 2); ImageNet 3x3 Kendall tau .73, tying a .72 baseline. C. elegans neuron identification, accuracy easy / hard at 40 % known neurons: .97 / .96 against .97 / .95 for Linderman et al. 2017 (Table 3). The ablation that matters for a port (Table 4, CelebA 5x5, Kendall tau): tau = 0.01 gives .62, tau = 100 gives .47, one Sinkhorn iteration gives .72, against .78 in Table 2. Nothing in the paper concerns lanes, maps, rectangular or partial matchings, or speed.

**Transferable technique** (no K id yet; neighbours: K145 PolyWorld Sinkhorn wiring, the SuperGlue extraction [C30](../../../deep/C30_superglue.md); register a new K id before use):
- Relaxation and training: log-domain Sinkhorn on the entry-to-exit score matrix S, tau = 1 (when S comes from a learned layer its scale makes tau redundant [judgement]; the paper treats tau as a searched hyper-parameter and does not state its values), 10 to 20 iterations (paper: 20, and one iteration costs only a little, Table 4), loss on log P. The Gumbel noise and the sample-averaged reconstruction loss are for latent-variable use; with labelled pairs a direct loss on P (negative log-likelihood of true pairs, as SuperGlue) replaces them.
- Rectangular, partial: pad S with one dustbin row and one dustbin column filled with a learnable score z and use marginals [1 per entry, X for the bin row] and [1 per exit, E for the bin column] (SuperGlue form, digest in C30). For real degrees above 1 (a fork, a merge) one-to-one is wrong: use capacities (row sums = out-degree, column sums = in-degree), because the 0/1 ground-truth matrix is exactly a transport plan with those marginals [derived].
- Sketch of the forward pass (NOT run; one junction, E entries, X exits, z a 0-dim tensor or parameter):
```python
def sinkhorn_dustbin(S, z, tau=1.0, iters=20):          # S: (E, X) entry->exit scores
    E, X = S.shape
    Z = torch.cat([torch.cat([S, z.expand(E, 1)], 1), z.expand(1, X + 1)], 0) / tau   # + bin column, bin row
    log_a = torch.cat([S.new_zeros(E), S.new_tensor([float(X)]).log()])  # row masses: 1 per entry, X for the bin
    log_b = torch.cat([S.new_zeros(X), S.new_tensor([float(E)]).log()])  # column masses: 1 per exit, E for the bin
    u, v = torch.zeros_like(log_a), torch.zeros_like(log_b)
    for _ in range(iters):                               # log-domain Sinkhorn-Knopp, columns last
        u = log_a - torch.logsumexp(Z + v[None, :], dim=1)
        v = log_b - torch.logsumexp(Z + u[:, None], dim=0)
    return Z + u[:, None] + v[None, :]                   # log P_bar (E+1, X+1); P = exp(.)[:E, :X]
```
- Plug-in: after the node features of the junction head (tokens for the entries and exits of one junction, plus heading and endpoint geometry), S_ij = bilinear(f_i, f_j) + a geometric term; P (soft) weights the edges that the assembly keeps; the only hard step is the named threshold on the output (D55). Interface, spec tests and first experiment: [port_card.md](port_card.md).

**Why it performs** (the paper's ablation): only Table 4 (tau and L at extreme values) is shown. The paper has no ablation of Sinkhorn against an independent row-softmax or sigmoid, so the gain from the doubly-stochastic competition is argued by the mechanism, not measured against that control. For SuperGlue the optimal-matching layer alone raises match precision from 50.4 to 66.0 over nearest neighbour plus mutual check (Table 4 of that paper, digest in C30); that is also not a control against independent pair scores.

**Cost and fit**: parameters one scalar z plus the score head; arithmetic E x X per iteration, so a junction of at most about 8 x 8 is launch-bound, not compute-bound: 2 logsumexp per iteration, about 40 small kernels at 20 iterations, well under a millisecond batched over all junctions of a frame [estimate, not measured; the paper gives no runtime or memory figure, checked]. Memory of the unrolled backward is 2L tensors of (B, E+1, X+1), negligible [derived]. Needs from us: the ground-truth pair matrix per junction (we have it: the oracle row of the brief), fixed padded shapes with masks, and a decision on one-to-one versus capacities. Effort S for the dustbin form, M for capacities.

**Doubts**:
- One-to-one is the paper's setting. A lane junction is often one-to-many (an entry that may turn left, go straight or turn right) and many-to-one (merges). Whether our ground-truth pairs are a partial permutation is unknown to me; the degree count in the port card decides it.
- All benchmarks are synthetic or small (numbers up to 120, jigsaws up to 36 pieces, up to 60 candidate neurons in Table 7), square, with the score row of an item computed from that item alone in the sorting network (App. B.2), not from a pair; no detection noise, no unmatched items.
- Optimiser, learning rate and the tau of each experiment were not found on the page; tau = 1 is stated only for Figure 2 and the Table 4 extremes are 0.01 and 100.
- Gradients vanish for small tau and the soft output is far from a vertex for large tau (Section 3, short quotes); the reference code warns that low tau needs more iterations and gives high-variance gradients (code page, checked). Hard rounding at inference (Hungarian in the paper) is not what is trained; a threshold on P needs tuning on VAL.
- What it does NOT answer: which features make an entry-exit pair plausible (the score function: TopoLogic-like geometry, learned tokens), how entries and exits of a junction are found, one-to-many degrees, calibration of P, consistency of pairs across several junctions, speed on a GPU, and whether competition beats independent pair scores at all.

**Sources opened** (2026-11-10): paper abstract https://arxiv.org/abs/1802.08665 (checked); paper full text https://arxiv.org/html/1802.08665 and https://ar5iv.labs.arxiv.org/html/1802.08665 (read through the fetch tool's summary: numbers are digest-grade, Table 4 agrees on the two renderings after one misread was caught); code https://github.com/google/gumbel_sinkhorn and its file sinkhorn_ops.py (checked); https://github.com/perrying/gumbel-sinkhorn, https://github.com/Thinklab-SJTU/ThinkMatch and its GumbelSinkhorn documentation and source page, https://github.com/Thinklab-SJTU/pygmtools, https://github.com/PythonOT/POT (checked, repository pages); https://arxiv.org/abs/1911.11763 (SuperGlue), https://arxiv.org/abs/2111.15491 (PolyWorld), https://arxiv.org/abs/2205.06688 (implicit Sinkhorn differentiation), abstract pages (checked). The SuperGlue equations come from the local extraction C30 (digest). Statements tagged [derived], [judgement], [estimate] are mine, not on any page.
