# RB22 - LearnableBN: test-time adaptation of BN for BEVFormer under corruptions

*Pass: loop 17 (2026-10-24), file `B_failure_robustness.md`, cluster `failure_robustness`. Depth M here means: method sections queried through a page digest (see sources/lit17/B_failure_robustness.md, section 1).*

- papers: Improving Batch Normalization with Test-Time Adaptation for Robust Object Detection in Self-Driving (Liao, Qi, Liu, Ma, arXiv 2411.18860; AAAI 2026 listing) https://arxiv.org/abs/2411.18860 (opened HTML v1); AAAI page https://ojs.aaai.org/index.php/AAAI/article/view/37626
- depth: M
- stage: T
- idea: Each BN layer gets one learnable mixing coefficient between source and test statistics; only these coefficients are optimised online with entropy minimisation plus a "generalised-search" term; a two-stage update (lr 2e-8 then 2e-7) keeps only samples whose KL between stages is in the lowest 10 % of history.
- training: no labels; backbone frozen; batch size 1 for nuScenes-C.
- differentiable: yes (update step at test time); output not affected otherwise.
- sensors_data: camera, BEVFormer R101, nuScenes-C six corruption types.
- results (average NDS-type score, Table 1): Low Light baseline 0.3212 -> LearnableBN 0.3469; Motion Blur 0.3330 -> 0.3787; Fog 0.4796 -> 0.4808. In the same table TENT 0.2070 / 0.2823 / 0.3323, AdaBN 0.0772 / 0.1279 / 0.1333, ARM(BN) 0.0961 / 0.1484 / 0.1431 are far BELOW the baseline at batch size 1. Headline gain: about 8 % (arXiv v1) or 10 % (AAAI version).
- compute: no runtime reported; one extra forward / backward over BN coefficients.
- code: not released at the time of v1 ("soon").
- fit_to_us: reuse_idea - effort S - warns that plain AdaBN / TENT can collapse in a per-frame BEV stream; a coefficient-mixing BN (source vs batch statistics) is the safe variant; our +0.04 with adaptive BN on the lane head may be fragile.
- tried_before: adaptive batch-norm (+0.04).
- verified: checked
