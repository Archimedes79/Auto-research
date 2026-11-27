# gumbel-sinkhorn: port card (nothing here was run: no shell in this session; all numbers marked [derived] are my algebra)

## 0. Free first step: count the ground-truth degrees (before any model)
From the stored ground truth (autogt output) take every junction with its entries (E) and exits (X) and the 0/1 pair matrix G (E x X). Report: the share of junctions in which every row sum and every column sum is at most 1 (a partial permutation: the paper's one-to-one setting with unmatched items), the histogram of out-degrees k_i and in-degrees m_j, and the share of entries without an exit and exits without an entry. The answer chooses the form: partial permutation -> dustbin form (section 1, Sinkhorn); many degrees above 1 -> capacities or a row-softmax with bin (method.md section 5). A threshold such as "90 % of junctions are partial permutations" is a proposal for the planner to fix before the run.

## 1. Interface (our code, not written)
```
JunctionAssign(d, geo_dim, tau=1.0, iters=20, per_item_bin=False)       # nn.Module, 1 scalar z (or Linear(d, 1) per item)
  forward(ent_tok (B,E,d), ent_geo (B,E,g), ent_ok (B,E) bool,
          exit_tok (B,X,d), exit_geo (B,X,g), exit_ok (B,X) bool) -> log_P (B,E+1,X+1)
      S_ij = <W_e ent_tok_i, W_x exit_tok_j> / sqrt(d) + MLP_geo([heading_i - heading_j, endpoint distance, ...])
      padded rows and columns: score -1e4 and mass 0 (log mass -1e4), bin masses use the true counts E_valid, X_valid
  P = log_P[:, :E, :X].exp()   pair probability;  log_P[:, :E, X] entry has no exit;  log_P[:, E, :X] exit has no entry
pair_nll(log_P, G (B,E,X) bool, ent_ok, exit_ok, ignore (B,E,X) bool = None) -> scalar
      minus the sum of log P_bar over true pairs, over entries with no exit, over exits with no entry (SuperGlue form, C30 Eq. 10)
```
Sketch of the iteration: [note.md](note.md). Rows are the nodes of one junction found by the lane head (entries = nodes where lanes enter the junction area, exits = where they leave; direction is known to be right in most cases per our records); the soft P multiplies the candidate edges in the assembly; the hard step is the threshold on P (and a mutual-maximum check if one-to-one is wanted), named as the output step (D55). Hungarian is not on the training path; scipy's solver (used by the reference code, CPU) is acceptable only as an offline score of the rounding.

## 2. Spec tests (toy cases with known answers; float64)
All with a function `sinkhorn_dustbin(S, z, tau, iters)` as in the note, or the same code without the bin for T1 and T5.
- **T1, 2 x 2 closed form [derived]**: Sinkhorn preserves the cross-ratio P11 P22 / (P12 P21) = K11 K22 / (K12 K21), and a doubly stochastic 2 x 2 matrix is [[p, 1-p], [1-p, p]], so p = sigmoid((S11 + S22 - S12 - S21) / (2 tau)). S = [[2, 0], [0, 2]], tau = 1, 100 iterations: P11 = sigmoid(2) = 0.880797, P12 = 0.119203 (tolerance 1e-9). Convergence bound for this matrix: tanh(1)^2 = 0.58 per iteration [derived from the Birkhoff contraction coefficient, memory], so 100 iterations are exact to machine precision and 20 iterations leave a bound of about 2e-5 (0.58^20).
- **T2, dustbin on 1 x 1 [derived]**: E = X = 1, matrix [[s, z], [z, z]] / tau with masses [1, 1] and [1, 1] is the same 2 x 2 problem: P11 = sigmoid((s - z) / (2 tau)). s = z gives 0.5; s - z = 2, tau = 1 gives sigmoid(1) = 0.731059, and the bin column, bin row (0.268941 each) and the corner (0.731059 = P11) follow.
- **T3, marginals**: random S of size 4 x 3, z = 0.3, tau = 1, 100 iterations: row sums of P_bar are [1, 1, 1, 1, 3], column sums [1, 1, 1, 4] (bin masses X = 3 and E = 4; total 7 on both sides), tolerance 1e-5 (columns exact because they are updated last).
- **T4, corner identity [derived]**: P_bar[E, X] equals the sum of the real block P_bar[:E, :X] (the bin row mass X minus the unmatched mass X - K), tolerance 1e-5.
- **T5, diagonal limit [derived]**: S = s I_N without bin, tau = 1: P = diag e^s / (e^s + N - 1) plus uniform 1 / (e^s + N - 1) off the diagonal, exact after one iteration (N = 3, s = 5: 0.986703).
- **T6, reference**: the log-domain torch result equals a plain numpy Sinkhorn-Knopp (`u = a / (K v); v = b / (K.T u)`, K = exp(S_bar / tau), P = u K v, 1000 iterations) to 1e-6; optional second reference `ot.sinkhorn` of POT with cost -S_bar, reg = tau, and masses divided by their sum then the plan multiplied back (POT needs an install: ask first).
- **T7, gradient**: `torch.autograd.gradcheck` of -log P_bar[0, 1] with respect to S and z in float64 on a 3 x 3 case with bin.
- **T8, the paper's own number (smoke test, optional)**: sorting U(0,1) with the paper's reconstruction loss gives zero sequences with an error at N = 5, 10, 15 (Table 1). Optimiser and learning rate are not on the page, so a faithful reproduction is not specified; use it only to check a training loop.

## 3. Cost
- Parameters: the bin score z (one scalar) plus the score head (two projections d x d_k and a small geometry MLP); nothing else.
- Compute per junction: 2 logsumexp per iteration over (E+1) x (X+1) entries; with E, X of about 8 at most it is launch-bound: about 2 x iters small kernels, batched over all junctions of a batch by padding [estimate: below 1 ms on the RTX 3060 with a captured CUDA graph, as in K186; must be measured on an idle GPU at batch 1]. The paper gives no timing.
- Memory of the unrolled backward: 2 x iters tensors of (B, E+1, X+1) [derived]: for B = 64 junctions of 9 x 9 and 20 iterations about 0.8 MB in float32. Implicit differentiation (https://arxiv.org/abs/2205.06688) is not needed at this size.
- Data: the pair matrix per junction (stored ground truth) and the entry / exit assignment of nodes; no new data.
- Effort: S for the dustbin form with the loss; M for capacities (degree head, or an upper bound) and calibration.

## 4. What to take, what to leave
- Take: the log-domain Sinkhorn iteration with the learned dustbin, tau fixed at 1 (scale lives in S), iterations as an ablation {1, 3, 10, 20} (paper: 20; one iteration only costs a little, Table 4), the pair negative log-likelihood, masks for padding.
- Take as an arm, not as default: Gumbel noise on the real block (noise_factor), per-item bin score.
- Leave: the latent-permutation reconstruction objective, the 10 x 10 sample batching, the KL / variational part, Hungarian in the training path, any library dependency (12 lines are re-implemented), the "score row from one item" network of the sorting task.

## 5. First experiment shape (proposal, no E id; the rule is for the planner to write)
Arms on the same junction features, same labels, 2 seeds each, VAL, paired scene bootstrap, noise band 0.010: A current learned pairs (junction F1 0.71, figure from the brief); B independent sigmoid pairs on the same score S, BCE (the control the paper lacks); C row softmax with bin column (first half-step of Sinkhorn); D Sinkhorn with dustbin, L = 20, tau = 1; E = D with L in {1, 3, 10}; F capacities (true degrees at training, a degree head or an upper bound at test). Reference rows from the brief: oracle pairs 0.88 and all pairs 0.53 (an oracle is a check, not a bound). Report junction F1, topo F1, ms per frame (idle GPU, batch 1), and the share of junctions where the rounding (threshold, mutual maximum) changes the pair set.

## 6. Risks
- One-to-one mismatch (section 0): the largest risk; the dustbin form then forces forks into the bin or splits mass 1/k over the exits.
- Gradient vanishing at low tau (paper Sec. 3; Table 4: tau = 0.01 gives Kendall tau .62 against .78) and high-variance gradients (reference docstring): keep tau at 1 and let S scale; do not anneal without a reason.
- Saturation: a confident wrong pair gives a very negative log P_bar; the loss is computed in the log domain (finite), clip it if needed.
- Convergence: after the last column update the row marginals are only approximate; end on the row update if rows are read as a categorical.
- Hard rounding at inference: P is soft at tau = 1; a threshold or mutual maximum is tuned on VAL; Hungarian forces one-to-one and is a CPU call.
- Padding: the reference code has no masks; bin masses must use the true E and X per junction.
- Order and symmetry: the score is pairwise, so the head sees both items; unlike the sorting network of the paper it does not produce a row from one item alone.
