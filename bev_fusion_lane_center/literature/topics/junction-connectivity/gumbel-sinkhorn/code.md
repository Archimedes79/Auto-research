# gumbel-sinkhorn: what the code offers

The paper page (https://arxiv.org/abs/1802.08665) links no code (checked: only generic code-finder toggles). Repositories opened as pages on 2026-11-10, nothing cloned or installed. All entries below are **checked** on the page named, unless marked.

| repository | what it is | licence | state | use for us |
|---|---|---|---|---|
| https://github.com/google/gumbel_sinkhorn | the reference code: TensorFlow 1.x, Bazel BUILD files; folders `sorting/`, files `optimizer.py`, `sinkhorn_ops.py`, `sinkhorn_ops_test.py`; no usage README (only Google's contribution guide). The README of the PyTorch port below links to it as Google's original implementation (page of that port, via the fetch summary), and two of the four authors were at Google at the time of the paper (memory) | Apache-2.0 | archived by the owner on 2022-09-07, read-only; 4 commits, 78 stars | read the 100-line `sinkhorn_ops.py` as the numeric reference; do not run (TF1) |
| https://github.com/perrying/gumbel-sinkhorn | PyTorch implementation of the paper: folders `jigsaw_puzzle/`, `number_sorting/`, `utils/`; `python train.py` in each experiment folder; needs torch, torchvision, numpy, scipy | not shown on the page | 4 commits, 33 stars, date not shown | reference for the sorting experiment (smoke test of a training loop); functions not visible on the page |
| https://github.com/Thinklab-SJTU/ThinkMatch | deep graph matching research protocol (883 stars); `src/lap_solvers/sinkhorn.py` holds `Sinkhorn`, `GumbelSinkhorn`, `BinSinkhorn` (source page read) | Mulan PSL v2 (first line of its LICENSE file) | 112 commits; the log-scale path calls pygmtools | idea level; `BinSinkhorn` is a dustbin form (see below); licence terms must be read before any reuse |
| https://github.com/Thinklab-SJTU/pygmtools | "A Python Graph Matching Toolkit", backends NumPy, PyTorch, Jittor, PaddlePaddle, TensorFlow, MindSpore; lists a differentiable soft Sinkhorn and an exact Hungarian solver | not shown on the page | 352 stars, 621 commits | numeric reference only |
| https://github.com/PythonOT/POT | Python Optimal Transport library, differentiable solvers with a PyTorch backend, entropic Sinkhorn-Knopp, stabilised and unbalanced variants | MIT | 2.9k stars | the numeric reference for a test of the generalised-marginal form (needs an install: ask first) |

## Reference code, read from the file `sinkhorn_ops.py` of the Google repository (page fetched through the summarising tool)
- `sample_gumbel(shape, eps=1e-20)`: uniform noise, double log.
- `sinkhorn(log_alpha, n_iters=20)`: batch of N x N matrices; each iteration subtracts a row-wise, then a column-wise `logsumexp`; exponentiates once at the end (log-domain, stable).
- `gumbel_sinkhorn(log_alpha, temp=1.0, n_samples=1, noise_factor=1.0, n_iters=20, squeeze=True)`: tiles `log_alpha` n_samples times, adds Gumbel noise times noise_factor (none if 0), divides by `temp`, calls `sinkhorn`; returns the matrices and the noisy scaled logits. The docstring warns that low temperature needs more iterations to avoid fractional entries and gives high-variance gradients.
- `matching(matrix_batch)`: `scipy` `linear_sum_assignment` through `tf.py_func` on the negated matrix: the Hungarian step, a CPU call outside the graph.
- Helpers: `invert_listperm`, `kendall_tau`, `sample_uniform_and_order`, `sample_permutations`, `permute_batch_split`, list / matrix permutation converters.

## ThinkMatch source page (https://thinkmatch.readthedocs.io/en/latest/_modules/src/lap_solvers/sinkhorn.html)
- `Sinkhorn(max_iter=10, tau=1., epsilon=1e-4, log_forward=True, batched_operation=False)`; the log path delegates to `pygmtools` (code not on the page); `nrows`, `ncols` give per-sample sizes; `dummy_row=True` pads zero rows to a square (padded entries epsilon), removed after the iterations.
- `GumbelSinkhorn`: wraps `Sinkhorn`, repeats each input `sample_num` (default 5) times, adds noise `-log(-log(u + eps) + eps)`, eps 1e-20; the documentation page says it only supports log-scale operation and that the original algorithm needs square matrices.
- `BinSinkhorn(tau=0.1, max_iter=50)`: log-domain updates `u = log_mu - logsumexp(Z + v)` and `v = log_nu - logsumexp(Z + u)`, a "prompt bucket" row and column appended to the scores: the same dustbin idea as SuperGlue, the closest ready-made form of our block.
- No documented `unmatchrows` / `unmatchcols` (searched on both pages).

## Reuse verdict
- Re-implement from the equations: the forward pass is about 12 lines (sketch in [note.md](note.md)); nothing needs the repositories.
- Take from the code: the log-domain pattern, the iteration default (20 in the paper code, 10 in ThinkMatch, 50 in its bin form), the warning that low temperature needs more iterations, a numeric reference for the test.
- Leave: TF1 code, Hungarian through `py_func` in the training path, sampling with `n_samples`, the sorting / jigsaw scripts.
- Third-party code is not committed to main (workspace rule); no licence problem arises when only equations are re-implemented.
