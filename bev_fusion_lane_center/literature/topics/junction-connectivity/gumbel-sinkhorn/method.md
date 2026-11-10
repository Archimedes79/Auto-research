# gumbel-sinkhorn: method, results, what is not shown

Read 2026-11-10. Marks: **paper** = arXiv HTML text of https://arxiv.org/html/1802.08665 and https://ar5iv.labs.arxiv.org/html/1802.08665, read through the page fetch tool (a summary of the page: numbers are digest-grade; tables were read twice and agree; the last 1341 characters of the first page, the reference list, were not read); **code** = repository page; **derived** / **judgement** = mine, on no page.

## 1. Setting (paper)
Latent variable model with a permutation P in the set of N x N permutation matrices. A network g maps the input X~ (a set of N items, in a scrambled order) to a score matrix X = g(X~, theta), N x N ("log alpha"). Training wants gradients through the choice of P; the hard matching M(X) = argmax over P of <P, X>_F has none.

Tensor shapes in the paper's networks (App. B.2, paper): sorting, one number per item, a first layer with N x n_u weights (n_u = 32 hidden units) and one row of N scores per number, so each row of X comes from one item alone, not from a pair; images, a convolution (K_s^2 n_c n_f + n_f parameters) and a layer giving N = p^2 outputs per piece. Largest model: 1,053,440 parameters (ImageNet 3x3, n_f = 64, s = 2, K_s = 5).

## 2. Equations (paper numbering where the page gave one)
- Sinkhorn operator (Sec. 2, Eq. 1; App. A Def. 1): S^0(X) = exp(X) elementwise; S^l(X) = T_c(T_r(S^{l-1}(X))) with T_r(X) = X (row sums)^-1 and T_c(X) = X (column sums)^-1; S(X) = lim_l S^l(X), doubly stochastic by Sinkhorn's theorem. In practice truncated at L; all experiments L = 20 (App. B.1); the L = 1 column of Table 4 shows a small drop.
- Matching operator (Eq. 2): M(X) = argmax_P <P, X>_F. Computed with the Hungarian algorithm (Munkres 1957) at test time (Sec. 3.1, App. B.1).
- Theorem 1 (Eqs. 3 and 4): S(X/tau) is the unique maximiser of <P, X>_F + tau h(P) over the Birkhoff polytope with h(P) = -sum P log P; and under an absolutely continuous i.i.d. distribution on the entries of X, M(X) = lim_{tau to 0+} S(X/tau) almost surely.
- Gumbel-Matching (Sec. 4): P ~ G.M.(X) when P = M(X + eps), eps an N x N matrix of i.i.d. standard Gumbel noise. Gumbel-Sinkhorn: P ~ G.S.(X, tau) when P = S((X + eps)/tau); samples tend to Gumbel-Matching samples as tau goes to 0.
- Variational use (Sec. 4.1, App. B.3): posterior G.S.(X, tau), prior G.S.(0, tau_prior), code variable Z = (X + eps)/tau, closed-form KL with the Euler constant. The page notes a possibly looser bound because S is not invertible. Not needed for supervised pairs.
- Supervised / reconstruction objective (Sec. 3, Eq. 5): f(theta, X, X~) = sum_i || X_i - P^-1(theta, X~_i) X~_i ||^2 with the hard P replaced by S(g(X~, theta)/tau); X_i the true ordered data of example i.

## 3. Schedule and hyper-parameters (paper, App. B.1 and Sec. 3 / 5)
- L = 20 Sinkhorn iterations; batch 10 x 10 (10 inputs x 10 Gumbel-perturbed reconstructions each).
- Noise: "always added to the output layer" as a regulariser and to make the matching unique (Theorem 1).
- Temperature: a hyper-parameter search over temperature, learning rate and network size is stated; the chosen tau per experiment is not on the page (tau = 1 in Figure 2 and 4 only). Too small: "gradients vanishes almost everywhere"; too large: S(X/tau) "may be far from the vertices" (Sec. 3).
- Optimiser, learning rate, number of steps: not found. Hardware: Tesla K20 / K40 / K80 / P100 cluster. Runtime and memory: not reported.
- Reference code defaults (code): `sinkhorn(log_alpha, n_iters=20)`, `gumbel_sinkhorn(log_alpha, temp=1.0, n_samples=1, noise_factor=1.0, n_iters=20)`; noise_factor 0 turns the noise off.

## 4. Results (paper; datasets and baselines named)
| experiment | metric | Gumbel-Sinkhorn | baseline |
|---|---|---|---|
| sorting U(0,1), N = 5 / 10 / 15 (Table 1) | proportion of sequences with any error | 0 / 0 / 0 (also 0 at N = 80 and 100, .01 at N = 120) | .06 / .43 / .9 (Vinyals et al. 2015) |
| sorting U(1000,1001) (Table 1) | same | 0 / 0 / .07 at N = 5 / 10 / 15 and 1 at N = 80 to 120 | - |
| jigsaw CelebA 2x2, 3x3, 4x4, 5x5 (Table 2) | Kendall tau | 1.0 / .96 / .88 / .78 | - |
| same | proportion of wrong pieces | 0 / .03 / .10 / .21 | random guess (n-1)/n |
| jigsaw ImageNet 3x3 (Table 2) | Kendall tau | .73 | .72 (Cruz et al. 2017) |
| jigsaw MNIST 2x2 to 6x6 (Table 2) | Kendall tau | 1.0 / .83 / .43 / .39 / .27 | - |
| C. elegans, 40 % / 30 % / 20 % / 10 % known neurons, easy (Table 3) | identification accuracy | .97 / .92 / .76 / .44 | Linderman et al. 2017 .97 / .90 / .77 / .39; MCMC .85 / .51 / .29 / .16 |
| CelebA 5x5, extreme settings (Table 4) | Kendall tau / proportion wrong | tau = 0.01: .62 / .36; tau = 100: .47 / .5; L = 1: .72 / .28 | main setting (Table 2): .78 / .21 |

Reading for us: the operator learns an exact permutation for small N (sorting), degrades with N and with hard inputs (U(1000,1001) fails from N = 15); the temperature has a usable middle range and one iteration is already reasonable "as long temperature is chosen in an appropriate range" (App. C.1).

## 5. Rectangular, partial and many-to-many: what changes
The paper treats only same-size sets (Sec. 3.2: "two sets of the same size"; no discussion of unequal sizes found). Constructions for our case:
1. **Generalised marginals** (standard Sinkhorn-Knopp, [derived]): for E entries and X exits take K = exp(S/tau), marginals a in R^E, b in R^X with sum a = sum b, iterate u = a / (K v), v = b / (K^T u); P = diag(u) K diag(v). The paper's operator is a = b = 1.
2. **Dustbin** (SuperGlue form, digest in [C30](../../../deep/C30_superglue.md) section 2.2; abstract page https://arxiv.org/abs/1911.11763 confirms "rejecting non-matchable points" inside a differentiable optimal transport): S_bar is (E+1) x (X+1), the bin row, bin column and corner hold one learnable scalar z, marginals a = [1_E; X], b = [1_X; E]. Then P_bar[i, X+1] = 1 - sum_j P_ij is the probability that entry i has no exit; the corner equals the total matched mass K [derived].
3. **Dummy padding without a learned bin** (library form): ThinkMatch `dummy_row` pads zero rows to a square, entries filled with a small epsilon and removed after the iterations (page read); no learnable "unmatched" score, so a ground-truth "no exit" has no output to train.
4. **Capacities for degrees above one** [derived]: the 0/1 ground-truth matrix G (E x X) has row sums k_i (out-degree) and column sums m_j (in-degree), so it is a feasible plan for marginals a = k, b = m; with these marginals a loss that asks P_ij = 1 on true pairs is satisfiable. Without them (marginals all 1) a loss on pair log-likelihood cannot reach 0 for k_i > 1 and drives the entry's mass into the bin. The degrees are unknown at inference: predict them (a head), bound them (inequality, upper capacity), or drop to a row-wise softmax with bin (the first half-step of Sinkhorn), see C30 section 5.3 for the same discussion on a map assignment.
5. **Gumbel noise with a bin** is not specified in the paper or in the code I read; a plain choice is noise on the real block only [judgement]. For supervised pairs the noise is a regulariser at best; test it only as an arm.

## 6. Training signal for pair labels
The paper's loss is a reconstruction loss (labels only through the ordered data). With labelled pairs the direct loss of SuperGlue applies (C30, Eq. 10): minus the sum of log P_bar over true pairs, over entries without an exit (bin column) and exits without an entry (bin row); ambiguous cases ignored. For degrees above one use capacities (item 4) or a soft target (1/k_i over the true exits of entry i; KL instead of NLL) [judgement].

## 7. Not shown in the paper (found nowhere on the pages read)
Unequal sizes, partial matchings, a dustbin; any comparison with independent pair scores (row softmax, sigmoid) or with a hard Hungarian layer inside the loop; temperature annealing; runtime and memory; implicit differentiation (a later paper does it: https://arxiv.org/abs/2205.06688, abstract: analytic gradients of the Sinkhorn layer, better gradient stability and less GPU memory than automatic differentiation); calibration of P; behaviour on real, noisy scores from a detector.
