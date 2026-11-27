# MH47 - Sparse top-k / top-1 routing with load balancing (Switch Transformer; noisy top-k gating)

*Pass: loop 16 (2026-10-23), cluster `multihyp` (multiple hypotheses, set prediction, matching-free losses). Reading depth: **M (Switch), not read (Shazeer)** (A abstract, M method, C code page). Verified: checked. Pipeline stage: S4, T.*

## Papers

Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity (Fedus, Zoph, Shazeer, JMLR 2022) https://arxiv.org/abs/2101.03961 (read via ar5iv; last 28 k characters not read); Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer (Shazeer et al., 2017) https://arxiv.org/abs/1701.06538 (not opened; described in the Switch paper)

## Idea

A router picks the highest-probability expert per token; the layer output is p_i(x) * E_i(x), so the gate value keeps the router differentiable although the choice is hard; an auxiliary loss keeps the load balanced and avoids collapse onto few experts.

## Formulation

p_i(x) = softmax(W_r x); output p_i(x) E_i(x) for the argmax expert; loss_aux = alpha * N * sum_i f_i * P_i, f_i = fraction of tokens dispatched to i, P_i = mean router probability, alpha = 1e-2; capacity = tokens/N * factor (1.0 - 1.25), overflow tokens pass by the residual (<1 % dropped). Shazeer: noisy top-k (k >= 2) with separate importance and load losses (from the Switch description). Instabilities: hard routing and bfloat16 in the router softmax; remedies: float32 router, initialisation scaled down 10x, expert dropout 0.4 in fine-tuning; Switch-XXL still sometimes unstable. - input / output: token features -> expert assignment.

## Training

end to end; regime `none`.

## Differentiable?

partial - the choice is constant, gradient through the gate value of the chosen expert.

## Sensors and data

C4 text, multilingual (mT5 comparison).

## Reported results

over 7x pre-training speed-up vs T5-Base at equal FLOPs per token; the page summary quotes steps 60 k and 450 k for a 64-expert Switch-Base against T5-Base, "about 7.5x" in steps (the direction of that sentence is ambiguous in the summary); dropped tokens typically < 1 %.

## Compute

sparse; routing cost small.

## Code

not checked

## Fit to our project

reuse_idea - effort S - the load-balancing term is a documented antidote to children / experts that never get used (slots collapsing), usable as "slot usage" regulariser over a batch of tips.

## Needs

batch-level statistics f_i, P_i over the b slots.

## Conflicts

forces uniform slot use, whereas forks are rare among tips - a uniform-usage prior is wrong for the second slot; use only as a weak term (the rate of forks per tip in our data was not looked up here).

## Tried in our experiments

none
