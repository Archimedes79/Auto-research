# iclr2021-imtl Towards Impartial Multi-task Learning (IMTL-L / IMTL-G; Liu, Li, Kuang, Xue, Chen, Yang, Liao, Zhang; ICLR 2021)

| | |
|---|---|
| source | ICLR 2021 poster https://iclr.cc/virtual/2021/poster/2894 (listing seen in a search result, not fetched); OpenReview PDF https://openreview.net/pdf/1641df474b8e0e2f7dd6c0dda99081b06fed400c.pdf (not fetched: the OpenReview page returned a browser-verification screen); text read on a mirror https://object.cloud.sdsc.edu/v1/AUTH_da4962d3368042ac8337e2dfdd3e7bf3/ml-papers-txt/ICLR/2021/towards_impartial_multitask_learning__ef10626e.txt. No arXiv id was found (none in the search results), hence the id used here. |
| fact status | checked on the text mirror (formulas, cost table, results); authors / venue digest (search result) |
| groups | F4: 3 (IMTL-L is our Kendall form; IMTL-G is the feature-level gradient balance used by FULLER 2307.16617) |
| code / licence | the paper says a unified codebase "will be publicly available", no URL given; IMTL is a weighting method in LibMTL (MIT, see 2308.12029) and IMTL-G / IMTL-L are aggregators / scalarizers in TorchJD (MIT, see 2406.16232) |
| verdict | context (the idea is taken by the papers above; this note holds the cost table) |

**What it does** [checked]: two parts. IMTL-L (loss balance): scale each loss by e^{s_t} and add -s_t, total sum_t (e^{s_t} L_t - s_t), both the network and s_t trained by gradient descent: this is Kendall's uncertainty weighting with s -> -s, and its optimum is s_t = -log L_t (a log-loss, see 2308.12029). IMTL-G (gradient balance): weights alpha_t on the task gradients with a closed form such that the aggregated gradient has equal projection on every task gradient: with u_t = g_t / ||g_t||, D = [g_1.g_2, ..., g_1.g_T], U = [u_1.u_2, ..., u_1.u_T], alpha_{2:T} = (g_1 U)(D U)^-1, alpha_1 = 1 - sum alpha_{t>=2}. To save cost the gradients are taken w.r.t. the shared FEATURE z instead of the shared parameters, so the backbone is backpropagated once with the combined feature gradient.

**Results that matter** [checked]: cost on Cityscapes in seconds per iteration: uniform 1.201, uncertainty weighting / GLS / IMTL-L about 1.20, GradNorm 1.742, MGDA 1.73-1.78, PCGrad 2.087, IMTL-G (feature level) 1.776, hybrid IMTL 1.795, IMTL-G with parameter-level gradients 2.769. Quality (their runs): NYUv2 mIoU / normal cosine / depth L1: single-task 56.82 / 0.8827 / 0.5097; uncertainty weighting 57.20 / - / 0.4400; IMTL-L 58.36 / 0.8864 / 0.4173; IMTL (hybrid) 58.85 / 0.8888 / 0.4215. Cityscapes semantic mIoU / instance / disparity L1: single-task 76.67 / 21.61 / 4.182; uncertainty weighting 74.91 / 16.43 / 2.895; IMTL 77.00 / 15.96 / 2.905. CelebA mean accuracy over 40 attributes: uniform 90.01, uncertainty 90.34, IMTL-L 90.54, IMTL 91.12.

**Transferable technique**: (1) our `task_weighting="kendall"` is IMTL-L under another sign: no new code, and the evidence for it is the 0.5-1 point gain over uniform / uncertainty weighting on NYUv2 and CelebA; (2) the feature-level gradient balance is the cheap way to get a gradient-based method on a trunk-heavy model: K head backward passes + one trunk pass (see the FULLER note for the four-group recipe).

**Cost and fit**: measured by the authors, +48 % per iteration (feature level) against equal weights on a 2-3 task Cityscapes / NYUv2 setup with heavy decoders; with light decoders on a heavy trunk the overhead should be smaller [memory: reasoning, not measured].

**Doubts**: ICLR 2021 numbers use their own baselines; the Cityscapes instance score of IMTL (15.96) is below single-task (21.61); no code; the cost table is for 2-3 tasks.
