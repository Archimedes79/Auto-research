import sys
R = "bev_fusion_lane_center/results/"
S4 = ("S4-HOLDS on both seeds by the written rule of the card (E372): topo F1 against the same head with B' frozen -0.0003 / +0.0049, against the staged champion "
      "+0.0002 / +0.0051, junction F1 +0.0021 / +0.0029")
miss = []


def sub(f, pairs):
    s = open(R + f, encoding="utf-8").read()
    for old, new in pairs:
        if s.count(old) != 1:
            miss.append((f, old[:60], s.count(old)))
            continue
        s = s.replace(old, new)
    open(R + f, "w", encoding="utf-8", newline="\n").write(s)


for rep, f_out, ch in (("E2E_Final_Report_2026-11-18", "parts/13_outlook.md", "chapter 10"),
                       ("E2E_Report_2026-11-18", "parts/08_outlook.md", "chapter 5")):
    p = rep + "/parts/"
    sub(p + "01_intro.md", [
        ("Not done: S4 (a joint fine-tune so that the loss reaches the lane head) has not run, there is no TEST number",
         f"S4 (a joint fine-tune so that the loss reaches the node head of the lane head) is measured: {S4}; the lead took it as the champion (D120, by the card's rule and D109, the end-to-end variant at a tie; the owner can overrule it; {ch}, section B9). Not done: there is no TEST number"),
        ("Marks labelled PENDING stand for what has not been measured yet (S4).", "Marks labelled PENDING stand for what has not been measured yet."),
    ])
    sub(p + "02_system.md", [
        ("S4 (E372) has not run: **the champion is stage S1 with the lane head B' frozen, without a past block and without a map block** (PENDING for S4).",
         f"S4 (E372) is measured: {S4}. By the card's rule and D109 (the end-to-end variant at a tie) the lead took it (D120): **the champion is the two-stage recipe, stage 1 = the head on the frozen cache (S1), stage 2 = the head and the node head of B' trained together through the token loss (S4), without a past block and without a map block**; the pyramid of B' stays frozen."),
        ("It is cached: no gradient reaches it (a trained-through B' is a later card, S4).",
         "In stage 1 it is cached and no gradient reaches it; in stage 2 (S4, E372, the champion since D120) its node head runs inside the training step and the loss of the token head reaches it, while B' keeps its own node loss (the head's loss was 0.19 - 0.40 of the updates of the node head)."),
        ("**Cost counted in every result:**",
         "**Stage 2 (S4, E372, the champion since D120).** 3000 joint steps, batch 8, learning rate 1e-4 for the head and the node head, per-group clip: `jlive_train.py --head-run e373_cb_s<S> --train-nodes 1 --live-inputs 1 --steps 3000 --bs 8 --lr 1e-4` (the token cache is replaced by the live output of B''s node head). Training step about 0.29 s against 0.24 s of the head alone. The inference network is the same as in stage 1 (B' + the head; only the weights of B''s node head changed), so ms per frame and parameters do not change by construction (not re-timed in the card). A new token cache from the s4 B' is needed before a head-only experiment builds on this champion.\n\n**Cost counted in every result:**"),
    ])
    sub(p + "04_experiments.md", [
        ("| S4 (E372) | joint fine-tune so that the loss reaches the lane head | PENDING: not run | PENDING |",
         "| S4 (E372) | joint fine-tune so that the loss reaches the node head of the lane head (arms z: B' frozen, b: B' on its own loss, s4) | topo F1 s4: 0.6682 / 0.6813; s4 - z -0.0003 / +0.0049, s4 - staged champion +0.0002 / +0.0051 | S4-HOLDS on both seeds (by hand, the card's rule v3.2); the champion since D120 |"),
    ])
    sub(p + f_out.split("/")[-1], [
        ("3. S4, a joint fine-tune that lets the loss reach the lane head (E372). PENDING.",
         "3. S4, a joint fine-tune that lets the loss reach the node head of the lane head (E372): done, S4-HOLDS on both seeds, the champion since D120; open from it: a new token cache from the s4 B' before a head-only experiment builds on it, and the pyramid of B' is still frozen."),
        ("(S4: the joint fine-tune with the lane head)", "(S4 is done for the node head: E372; the pyramid stays frozen)"),
    ])

p = "E2E_Final_Report_2026-11-18/parts/"
sub(p + "05_status.md", [
    ("and a lane head that the loss does not reach (staged; S4 is the card that changes this, section 6). Status of this chapter as of the build: PENDING S4 verdict.",
     "and the pyramid of the lane head, which the loss does not reach (the node head of B' is trained in the loop since D120: two stages). Status of this chapter: S4 measured (E372, S4-HOLDS), the champion since D120."),
    ("| staged training | B' frozen, nodes cached | frozen | **named, kept** until S4 (E372) has a verdict: the loss of the token head does not reach B' |",
     "| two-stage training | B' frozen, nodes cached | frozen | **two stages (D120):** stage 1 on the frozen cache, stage 2 trains the node head of B' with the head (the loss reaches the node head); the pyramid stays frozen |"),
    ("except that the lane head B' is trained earlier on its own loss.",
     "except that the pyramid of the lane head B' is frozen (trained earlier on its own loss) and its node head keeps that loss in stage 2 (the champion is trained in two stages)."),
])
s = open(R + p + "05_status.md", encoding="utf-8").read()
i = s.index("**S4, the step that closes the staged gap (E372).**")
s = s[:i] + ("**S4, the step that closes the staged gap (E372): measured, S4-HOLDS on both seeds, the champion since D120.** The node head of B' runs inside the training step, the loss of the token head reaches it, and B' keeps its own loss so that it does not drift; three arms (z: the same fine-tune with B' frozen; b: B' on its own loss only; s4), two seeds, VAL 873 frames, topo F1 read live at the D115 cell: z 0.6685 / 0.6763, b 0.6663 / 0.6787, s4 0.6682 / 0.6813, the staged champion 0.6679 / 0.6762. s4 - z -0.0003 / +0.0049 (the interval of seed 1, [+0.0020, +0.0079], is above 0, that of seed 0 is not), s4 - staged champion +0.0002 / +0.0051; junction F1 +0.0021 / +0.0029; the loss of the head made 0.19 - 0.40 of the updates of the node head, the tokens moved on every frame (median 5 cm). By the written rule of the card the verdict is S4-HOLDS (quality kept, the gradient reaches the node head), not S4-GAINS (it needs s4 - z of at least +0.01 on both seeds); the lead took it as the champion (D120, with D109: the end-to-end variant at a tie).\n")
open(R + p + "05_status.md", "w", encoding="utf-8", newline="\n").write(s)
sub(p + "06_challengers.md", [
    ("| S4, E372 | the loss reaches the lane head (joint fine-tune) | PENDING: the rule of the card (arms z, b, s4) | PENDING; interim seed 0, live read-out: z 0.6685, s4 0.6682, champion 0.6679 | PENDING | E372 |",
     "| S4, E372 | the loss reaches the node head of the lane head (joint fine-tune; arms z, b, s4) | S4-HOLDS: s4 - z not lower than -0.01 in topo F1 and junction F1, s4 - ref not lower than -0.01, floors, parity; S4-GAINS needs s4 - z >= +0.01 on both seeds | topo F1 0.6682 / 0.6813; s4 - z -0.0003 / +0.0049; s4 - ref +0.0002 / +0.0051 | **S4-HOLDS, not S4-GAINS: the champion since D120** | E372, D120 |"),
    ("| champion (E373, E375) |", "| staged champion (E373, E375), the reference of S4 |"),
    ("| **the champion** | E373, E375, D115 |", "| the champion until D120 | E373, E375, D115 |"),
    ("The champion (the E365 token head, none weight 0.25, decode D115) stays unless a challenger beats it by its rule.",
     "The champion since D120 is the two-stage recipe (the E365 token head with none weight 0.25 and the decode D115 as stage 1, then S4: the head and the node head of B' trained together); a challenger replaces it only by its rule."),
])
print("misses:", miss)
