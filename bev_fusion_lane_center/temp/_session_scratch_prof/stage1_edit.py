R = "bev_fusion_lane_center/results/E2E_Report_2026-11-18/parts/"
miss = []


def sub(f, pairs):
    s = open(R + f, encoding="utf-8").read()
    for a, b in pairs:
        if s.count(a) != 1:
            miss.append((f, a[:60], s.count(a)))
            continue
        s = s.replace(a, b)
    open(R + f, "w", encoding="utf-8", newline="\n").write(s)


# ---- 04 experiments: rows after the S4 row
s4row_start = "| S4 (E372) |"
s = open(R + "04_experiments.md", encoding="utf-8").read()
i = s.index(s4row_start)
j = s.index("\n", i)
rows = ("\n| E378 | the max-positive loss of E376 on two fresh heads (champion recipe otherwise) | topo F1 +0.0030 / +0.0000 at the recipe cell; junction F1 +0.038 / +0.012; x_s1 lanes found -0.0106 | NO (clause i fails on both heads; not a TIE either); the champion keeps the summed likelihood |"
        "\n| E379 | distilling a two-head ensemble into one head (KL to the teachers' mean successor distribution), and the plain two-head ensemble e2 | d at D115: +0.0074 / +0.0027 topo F1 (65 % / 42 % of the ensemble gain), at the recipe cell +0.0038 / +0.0008 with lanes found -0.0120 / -0.0171; e2 +0.0113 / +0.0065 at 21.84 ms | d: NO by the letter, TIE in substance; e2: MISS on cost (19.7 ms gate); the champion keeps one head |"
        "\n| E384 | a recursive node state in the node head of B' (step 0 on 851 VAL frames, no training) | oracle coverage ceiling +0.0021 .. +0.0037 points, AP lift +0.0017 / +0.0018 | CUT by hand at step 0 (critic step 0; the bars were written after the numbers); card kept ready |")
s = s[:j] + rows + s[j:]
open(R + "04_experiments.md", "w", encoding="utf-8", newline="\n").write(s)

# ---- 05 quant: sections B8, B9, B10 before C
B = """## B8. A fresh-head confirmation of the max-positive loss (E378, NO)

**The question.** E376 found that the max-positive likelihood (`--pos-loss max`: the loss of a source asks for ONE winner among the duplicate targets) lifts the junction F1 and perhaps the topo F1 on two heads. E378 asks whether that replicates on two FRESH head seeds, because the E376 heads were the ones where the effect had been seen. The champion recipe otherwise (none weight 0.25, 12000 steps), VAL 873 frames, read at the recipe cell (the cell chosen on the TRAIN sample by the card's rule), the rule written before the run: clause (i) topo F1 at least +0.005 with the interval above 0 on both heads; junction F1, F1@1.0 and lanes found not lower than the card's guards (0, 0.005, 0.010).

**Result (by hand, the card's rule).** NO at clause (i), the seed twins were not trained (the card trains them only after (i) passes). Topo F1 against the champion head of the same cache: x_s0 +0.0030 [-0.0023, +0.0108], x_s1 +0.0000 [-0.0046, +0.0051]; the junction F1 rises clearly at the recipe cell (+0.038 / +0.012) but x_s1 loses lanes found (-0.0106, guard -0.010), so the result is not a TIE either. The champion keeps the summed likelihood.

Table: The four max-positive heads at the D115 cell against the champion head of the same cache (by hand: stored numbers from the cards named in E378's section, not a verdict; the two e367_b25 rows are the champion recipe on another head seed, i.e. the noise of the same comparison).

| head (loss) | junction F1 [95 %] | topo F1 | F1@1.0 | lanes found |
|---|---|---|---|---|
| e376_mx_s0 (max) | +0.0158 [+0.0020, +0.0332] | +0.0055 | +0.0047 | +0.0097 |
| e376_mx_s1 (max) | +0.0078 [+0.0006, +0.0150] | +0.0110 | +0.0040 | +0.0107 |
| e378_x_s0 (max, fresh) | +0.0219 [+0.0062, +0.0452] | +0.0067 | +0.0054 | +0.0108 |
| e378_x_s1 (max, fresh) | -0.0016 [-0.0099, +0.0064] | +0.0015 | +0.0007 | +0.0011 |
| mean of the four max heads (SD) | +0.0110 (0.0102) | +0.0062 (0.0039) | +0.0037 (0.0021) | +0.0081 (0.0047) |
| e367_b25_s0 (sum, noise) | +0.0063 [-0.0039, +0.0181] | +0.0030 | +0.0016 | +0.0025 |
| e367_b25_s1 (sum, noise) | +0.0014 [-0.0050, +0.0098] | -0.0032 | -0.0006 | -0.0024 |

Read honestly: at D115 the max loss moves the junction F1 by about +0.011 and the topo F1 by about +0.006 on average over four heads, against a head-seed noise of about +-0.004 (junction) and +-0.003 (topo); one fresh head (x_s1) shows almost nothing. The junction gain at the recipe cell (+0.038) was paid in lanes found. A fresh confirmation pair at D115 with a junction bar of +0.015 was estimated at 10 - 15 % to pass and was not run.

## B9. Stage S4 (E372): the loss reaches the node head of the lane head (S4-HOLDS)

**The question.** The champion head is trained on tokens that a frozen lane head B' made once (a cache). S4 lets B''s node head run inside the training step, so that the loss of the token head reaches it, while B' keeps its own node loss so that it does not drift (the pyramid stays frozen). Three arms, each on two seeds, VAL 873 frames, read live at the D115 cell: z (the same fine-tune with B' frozen: the control of the extra training), b (B''s node head on its own loss only), s4 (on both losses). The rule of the card (v3.2, written before any run; by hand, because `decide.py` has no cards listed for it): S4-HOLDS = s4 - z not lower than -0.01 in topo F1 and junction F1, s4 against the champion read live not lower than -0.01, the D113 floors, the parity rule; S4-GAINS needs s4 - z of at least +0.01 on both seeds with the interval above 0; "the head's gradient costs" = s4 - b of at most -0.01 on both seeds.

Table: S4 on VAL (873 frames, D115 cell, topo F1 / junction F1 / F1@1.0 / lanes found; ref = the champion read live with the untouched B').

| arm | seed 0 | seed 1 |
|---|---|---|
| ref | 0.6679 / 0.2517 / 0.8459 / 0.7191 | 0.6762 / 0.2452 / 0.8467 / 0.7205 |
| z (head fine-tuned live, B' frozen) | 0.6685 / 0.2553 / 0.8460 / 0.7200 | 0.6763 / 0.2454 / 0.8467 / 0.7200 |
| b (B''s node head on its own loss) | 0.6663 / 0.2516 / 0.8447 / 0.7200 | 0.6787 / 0.2467 / 0.8455 / 0.7199 |
| s4 (B''s node head on both losses) | 0.6682 / 0.2538 / 0.8459 / 0.7232 | 0.6813 / 0.2481 / 0.8476 / 0.7214 |
| s4 - z (topo F1, paired 95 %) | -0.0003 [-0.0043, +0.0036] | +0.0049 [+0.0020, +0.0079] |
| s4 - ref (topo F1, paired 95 %) | +0.0002 [-0.0042, +0.0043] | +0.0051 [+0.0019, +0.0083] |
| s4 - ref (junction F1, paired 95 %) | +0.0021 [-0.0037, +0.0101] | +0.0029 [-0.0018, +0.0084] |
| s4 - ref (lanes found, paired 95 %) | +0.0041 [+0.0002, +0.0083] | +0.0009 [-0.0037, +0.0054] |

**Result.** S4-HOLDS on both seeds, not S4-GAINS: the quality is kept while the loss reaches the node head (s4 - z -0.0003 / +0.0049; the interval of seed 1 is above 0, that of seed 0 is not; S4-GAINS needs +0.01). The gradient of the head does not hurt: s4 - b is +0.0019 / +0.0026 (two cards, no paired interval). The checks of the card: the parity rule passed exactly on both seeds (873 / 873 token sets equal, the cache read-out equal to the live read), B''s node head is proven s4's own, no kill (all arms finished 3000 steps, no roll-back), the head's loss made 0.19 - 0.40 of the updates of the node head, the tokens moved on every frame (median 5 cm), no moving-label flag. By D109 (the end-to-end variant at a tie) and the card the lead took it as the champion (D120): the recipe has two stages (the head on the frozen cache, then 3000 joint steps), the inference network is the same (16.6 - 16.8 ms, 2.323 M parameters; not re-timed in the card), the training step about 0.29 s against 0.24 s. A new token cache from the s4 B' is needed before a head-only experiment builds on it. VAL only, TEST untouched.

## B10. Distilling a two-head ensemble into one head (E379: d NO / TIE in substance, e2 MISS on cost)

**The question.** Two heads of the champion recipe averaged (the ensemble e2) give a better topo F1 than one; can that gain be put into ONE head at inference (d: the GT loss plus the KL of the successor distributions, including the none slot, to the mean of two frozen teachers, weight 1; d4 weight 4 on one seed as an exploration), or does the ensemble itself pass the cost gate of the card (the whole path at most the 19.7 ms of the previous system)? Rule written before the run: topo F1 at least +0.005 with the interval above 0 on both heads at the recipe cell, guards on junction F1, F1@1.0, lanes found and duplicates; the cost gate for e2.

**Result (the card's section 4, by the lead's reading of the written rule).** d at the D115 cell: topo F1 0.6753 / 0.6789 against the champion heads 0.6679 / 0.6762 (+0.0074 [+0.0035, +0.0114] / +0.0027 [-0.0020, +0.0072]), the ensemble e2 0.6792 / 0.6827 (+0.0113 / +0.0065): d recovers 65 % / 42 % of the ensemble gain with one head at inference. At the recipe cell of the card (the cell of the mean of the two TRAIN tables, 0.3 / 0.2 / 0.5) clause (i) fails on both seeds (+0.0038 [-0.0002, +0.0078] / +0.0008 [-0.0044, +0.0055]) and the lanes-found guard is broken on both (-0.0120 / -0.0171): NO by the letter; at the champion's own cell no guard breaks and clause (i) holds on seed 0 only: a TIE in substance. What d does on both seeds at both cells: junction F1 +0.008 .. +0.028, duplicates -0.007 .. -0.0125, a higher long-link AP (seed 0: 0.283 against 0.269). KD weight 4 gives the same as weight 1 (+0.0085, seed 0). The champion keeps one head and no teacher; the controls l and ds (a schedule control and a self-teacher) were not run, as the card says (only if d passes clause (i) on both seeds). e2 is a MISS on cost: 21.84 ms in sequence, 25.19 ms stacked, against 19.7 ms (section D).

"""
s = open(R + "05_quant.md", encoding="utf-8").read()
k = "## C. The oracle-node stage"
assert s.count(k) == 1
s = s.replace(k, B + k)
d_old = "Timing protocols are never mixed: only paired differences inside one protocol are compared."
assert s.count(d_old) == 1
d_new = """Table: What the pieces cost at inference, measured quiet in the night of 2026-11-18 (batch 1, bf16, RTX 3060, 24 frames x 5 repeats, the median of the timed frames, load below 3 and GPU idle before and after; `experiments/E2E_cost_table.md`).

| piece | median ms | parameters | against the champion path |
|---|---|---|---|
| champion path (B' + one head) | 16.21 | 2.323 M | - |
| e2: two heads in sequence | 21.84 | 2.450 M | +5.63 ms (+35 %), above the 19.7 ms gate of E379 |
| e2: two heads stacked (`vmap`) | 25.19 | 2.450 M | +8.98 ms (the head is latency-bound at batch 1) |
| S2 block (cross-attention to the past), head alone | 7.88 | 144,547 (+17,416) | +1.23 ms over the head without it (6.66) |
| S3 block (cross-attention to the SD map), head alone | 7.95 | 147,395 (+20,264) | +1.70 ms over the head without it (6.26) |
| S4 (the node head trained in the loop) | not timed | the same modules | the same network with other weights |

The first timings of the same night were flagged not quiet only because the tool took the GPU after-sample without the 2.5 s wait of the protocol; the repeat with the wait reads quiet and agrees within 0.2 ms. **K = 384 (a diagnostic of E383, no verdict):** the inference whole path 15.47 ms against 16.24 ms at K = 512 (the tool flagged both NOT QUIET by the same artefact), the training step 0.634 times K 512 (57.6 ms against 90.9 ms in one process); the VAL GT-node coverage at K 384 is 0.9878 / 0.9847 against the written 0.99, so K 384 is not taken. **Training step:** a profile showed the step of the head is GPU-bound (a whole-step CUDA graph gains 13 %); an exact speed path (a GPU-resident loader, a loss without host syncs, scales only where the loss reads them) was built and tested (same loss and gradients, GPU tests green, twin loss curves within 0.0001) and gained 1.071 times against the written gate of 1.3 (plain 98.8 ms, fast 92.3 ms per step): MISS, no parity run, not adopted (E383). The CPU scorers are the better lever: `jtoken_eval.py sweep --jobs N` scores the 36 cells in N processes, 2.5 times with N = 3 and identical numbers.

""" + d_old
s = s.replace(d_old, d_new)
open(R + "05_quant.md", "w", encoding="utf-8", newline="\n").write(s)

# ---- 08 outlook
sub("08_outlook.md", [
    ("5. The seed-twin interval and the frame-order check of E373 (CPU, open);",
     "5. E378 (the max-positive loss on fresh heads): NO (chapter 5, section B8); E379 (distilling a two-head ensemble): d NO / TIE in substance, e2 MISS on cost (section B10); E383 (a faster training step): MISS at its gate, not adopted (section D); E384 (a recursive node state in B', planned under D118): measured at step 0 on 851 VAL frames and closed by the oracle ceiling (+0.0021 .. +0.0037 coverage points against the bar +0.006, AP lift +0.0017 / +0.0018 against +0.005, 1 true to 8 false carried-over tokens): recursive temporal fusion without more input frames is planned and measured, not trained; S4 is the E2E part that was trained. The seed-twin interval and the frame-order check of E373 (CPU, open);"),
])
# ---- 07 winners
sub("07_winners.md", [
    ("| training stages | three, plus the attachment | one (the head; B' is frozen) |",
     "| training stages | three, plus the attachment | two (D120): stage 1 the head on the frozen cache, stage 2 the head and the node head of B' together (3000 steps, E372) |"),
])
print("misses:", miss)
