# 5. End-to-end status: what is differentiable and which hard steps are named

**In short.** The delivered graph of the champion is one network read out by thresholds: the frozen lane head B' makes the tokens, one token head with three self-attention layers wires them (a softmax per source with a none slot) and types them (a softmax over seven classes), and a loss on the output reaches the head. Five hard steps of the previous end-to-end system (E324 s) are gone; what remains is named below and is **not** hidden: a compute budget, a gate, four output thresholds and a lane head that the loss does not reach (staged; S4 is the card that changes this, section 6). Status of this chapter as of the build: PENDING S4 verdict.

Table: The named hard steps (E2E gap table of D109, rows R1 - R8, T1 - T3) and where each stands in the token champion. "Gone" = the step is not in the delivered path of the champion and nothing replaced it by a rule.

| step | what it was | in the previous E2E system (E324 s) | in the token champion |
|---|---|---|---|
| R1 snap wiring | joins each connector end to the nearest piece end within 4 m and 45 degrees | replaced by a learned attachment (E320); the snap geometry is only in the record read-out | **gone:** the wiring is the softmax of the last attention layer |
| R2 close-node step 1.5 m | drops a node near a node with higher existence | the soft Matrix-NMS layer of B' (E292) | **gone:** the none slot and p_none prune softly; B' keeps its soft layer |
| R3 candidate selection of B | top-K 512 nodes at e > 0.05, window and cone limits for long links | present | **named, kept:** the top-K 512 of the tokens by B''s existence is a compute budget, no gradient goes through it |
| R4 section argmax, one link per pair | hard choices on the delivered graph | present | **replaced by the decode:** a link is delivered when its probability passes a threshold; no argmax per section |
| R5 keypoint peaks and top-K 32 | 3 x 3 max-pool and a peak minimum select the ends | present | **gone** (E347 closed the grid-end variant at stage 1) |
| R6 pair candidate mask | a pair may connect only when turn angle and chord conditions hold | present | **named, kept:** the allowed-pair gate of the attention (`jtok.pair_geometry`); with K = 512 it lets 94.3 % of the GT links through (E365 manifest) |
| R7 attach gate | the attachment sees the nearest 8 nodes within 4 m and 45 degrees | present | **gone:** there is no attachment |
| R8 group and rank bookkeeping | the order rule that gives the pair head its features | present, label side | label side only (`jgt`, by design no gradient, D82) |
| T1 - T3 thresholds | node e >= 0.5, link p >= 0.5, connector p >= 0.1 | present | **four named output thresholds:** keep 0.6, link short 0.3, link long 0.2, none gate off (D115); class T stays by design (D55 / D88) |
| staged training | B' frozen, nodes cached | frozen | **named, kept** until S4 (E372) has a verdict: the loss of the token head does not reach B' |

**What "end to end" means here, and what it does not.** From the cached camera-and-radar features to the graph, every learned step is differentiable and trained on a loss at the graph level, except that the lane head B' is trained earlier on its own loss. The three hard items that remain in the forward path are the top-K, the gate and the thresholds of the read-out; each has a price that is measured or named (sections 7 and 13). A rule step in the delivered graph is reported every time: there is none besides these.

**S4, the step that closes the staged gap (E372).** The node head of B' runs inside the training step, the loss of the token head reaches it, and B' keeps its own loss so that it does not drift; two arms against a matched fine-tune without the gradient (z) and B' on its own loss only (b), two seeds, VAL. Interim, labelled as such (seed 0, live read-out at the champion's decode, from the lead's message of the night, not yet on a committed card: replaced by the card's numbers at the build; before the written rule of the card was applied to all arms): topo F1 z 0.6685, s4 0.6682, champion 0.6679. PENDING: the verdict by the written rule of the card after all arms.
