p = "experiments/E344_e2e-completion-plan/experiment.md"
s = open(p, encoding="utf-8").read()


def rep(a, b, cnt=1):
    global s
    assert s.count(a) == cnt, (s.count(a), a[:80])
    s = s.replace(a, b)


def between(start, end, new):
    global s
    i = s.index(start)
    j = s.index(end)
    s = s[:i] + new + s[j:]


# ---- header
rep("Plan of 2026-11-14 17:30, written by", "Plan of 2026-11-14 17:30, revised 19:45 after the first critic pass (REVISE: the count of selections, the connector representation that E273 measured, how a successor distribution meets B's walk, prior art E271 / E273 / E292, the gates, the costs, the defaults), written by")
rep("**Status: plan only; nothing built, nothing run.** Needs Mirko's answers to section 7 and a fresh `ar-critic` (cards) before the first line of code (D84 point 6 parked the \"unstructured transformer head\"; this plan proposes a grid-anchored variant of it and asks whether that is wanted).",
    "**Status: plan only; nothing built, nothing run. Q0 (CPU, ground truth only, card E347) may start under D86; Q1 and later wait for a recorded decision D<n> of Mirko on D84 point 6 and on the replacement of B's heads (section 7 holds the lane chat's reading, labelled as such).**")

# ---- JSON card
rep("\"title\": \"plan: remove the named output selections of the delivered graph (snap / wiring, threshold, top-K, close-node step, Douglas-Peucker) one by one and, as the end state, replace the keypoint / pair / connector chain and B's node / link heads by one query head anchored on the 1 m grid; the plan card's metric is the number of named selections, the guard is the accuracy\",",
    "\"title\": \"plan: reduce the named RULE steps of the delivered graph (counted in section 1: eight at E324 s) and, as the end state, replace the keypoint / pair / connector chain and B's node / link / section heads by one query head anchored on the 1 m grid; the plan card's metric is the number of named rule steps (consumer thresholds are named, not counted), the guard is the accuracy\",")
rep("\"metric\": \"named_output_selections\",", "\"metric\": \"named_rule_steps\",", cnt=2)
rep("\"prediction\": {\"value\": 0, \"vs\": \"c\"},", "\"prediction\": {\"value\": -6, \"vs\": \"c\"},")
rep("\"c\": {\"recipe\": \"control = E324 s (D94): five named output selections (snap and wiring of the connector ends, probability threshold p >= 0.1, top-K = 32 keypoints, close-node step 1.5 m, Douglas-Peucker 0.3 m), plus the group / rank bookkeeping (named, no gradient) and the frozen sensor features (D50)\"},",
    "\"c\": {\"recipe\": \"control = E324 s (D94): eight named rule steps R1 - R8 in the delivered graph (section 1: snap / nearest-piece wiring, close-node step, B's candidate selection, section argmax and link rules, keypoint peak choice and top-K = 32, pair candidate mask, attach gate, group / rank bookkeeping), three consumer thresholds T (node e >= 0.5, link p >= 0.5, connector p >= 0.1) and the frozen sensor features (D50)\"},")
rep("\"q\": {\"recipe\": \"Q-Lane at stage Q5 (section 4): the delivered graph is a probabilistic graph (nodes with existence, type, position mean and sigma, heading; edges with probabilities); the only selection left is the consumer's threshold\"}",
    "\"q\": {\"recipe\": \"Q-Lane at stage Q5 (section 4): the delivered graph is a probabilistic graph (nodes with existence, type, position mean and sigma, heading; edges with probabilities); at most two rule steps are left (the active set of the walk and the peak-choice equivalent, named); the thresholds T stay because scoring needs them (D55 / D88 ask for a soft training path and NAMED output selections, not for zero)\"}")
rep("{\"metric\": \"named_rule_steps\", \"vs\": \"c\", \"min_effect\": 1},", "{\"metric\": \"named_rule_steps\", \"vs\": \"c\", \"min_effect\": 4},")
rep("every stage of section 4 gets its own card with a rule written before the run.", "every stage of section 4 gets its own card with a rule written before the run, a predicted number, a paired scene bootstrap and an equal-step control (the pair-stage noise has never been measured: E318 revision item 1).")

# ---- section 1 and 1b
between("## 1. What \"end-to-end\" means here, and where we stand", "## 2. Principle", """## 1. What "end-to-end" means here, and where we stand (counted from the code, 2026-11-14)
The rule (D55, D88): everything on the training path is soft, a loss on the output reaches every module (checked with `evaluation/drivers/e2e_gradient_report.py`, "no silent tensor"), hard selection only on the delivered output and named in every report. D55 / D88 do NOT ask for zero selections: scoring needs thresholds on existence and on edges, so the end state is "no rule step on the path, the consumer's thresholds named". Three classes: **R** = a hand-written rule step or index choice in the delivered graph, **T** = a consumer threshold on a delivered probability (stays by design), **S** = the frozen sensors. The delivered graph of E324 s (snapped read-out) has:
| # | class | step | where | what it carries (VAL) / soft replacement | state |
|---|---|---|---|---|---|
| R1 | R | snap + nearest-piece wiring of the connector ends, with the end geometry (E311) | `jwire.py`, `qends_eval.add_connectors` | about two thirds of the topology gain (69 % / 73 % of it is the choice of ONE nearest piece). E324 s own keypoints snapped: 0.6981 / 0.7010 = +0.0156 / +0.0145 over E312; the soft-attachment read-out PRIMARY (attached ends, nearest-piece wiring, no snap geometry) 0.6912 / 0.6948 = +0.0087 / +0.0084, i.e. 0.0069 / 0.0062 below the snapped one (E324 section 5). E325 learns the wiring but ends in a head argmax, and its weights are not edge probabilities (E325 l.46, l.51) | geometry soft (E320 / E324), wiring still a rule |
| R2 | R | close-node step 1.5 m (E275) | `qgraph_readout` | B's plain graph topo F1 0.597 -> 0.626 on VAL, duplicates down; soft step E279 (trains the pyramid jointly, invalidates the junction heads), E288 Matrix-NMS (TIE), E292 layer (not adopted) | named |
| R3 | R | B's active-node and link-candidate selection: top-K = 512 with threshold 0.05, +-4-cell window, top-64 x top-64 long links, 8 m < d <= 100 m, 75 degree cone, a random cap of 80 000 pairs | `qlinks.py:82-126` | index choices of the soft walk | named |
| R4 | R | section argmax (`qpredict.py:38`), one link per node pair (`qgraph_eval.py:64-76`), long-link cover drop in the read-out (`jwire.py:21-36`) | read-out | hard choices on the delivered graph | named |
| R5 | R | keypoint peak choice (3 x 3 max-pool + PEAK_MIN 0.1, `jends.py:162-178`) and top-K = 32 | pair head | the peak choice is the real end suppression; **K binds in 16 and 23 of 873 VAL frames (1.8 % / 2.6 %, 12.1 / 12.5 valid ends per kind, measured on `runs/e312_p_s{0,1}/ends_val.pkl`)**: an index choice, not an accuracy cap | deferred since D84 |
| R6 | R | pair candidate mask `allowed` (`jends.py:220`) | pair head | which pairs may connect | named |
| R7 | R | attach gate: nearest 8 nodes within 4 m / 45 degrees with e > 0.05 (`jends.py:117-121`) | E320 / E324 s attachment | candidate set of the soft attachment | named |
| R8 | R | group / rank bookkeeping of the keypoints (order rule, `jgt._groups`, `_ranks`) | pair head | features of the pair head, no gradient by design (D82); E342 vectorises it | named |
| T1-T3 | T | node e >= 0.5, link p >= 0.5, connector p >= 0.1 | read-out | 0.5 -> 0.1 on the connectors gives lanes found +0.0616 / +0.0656 snapped and +0.0625 / +0.0688 unsnapped on E312 (`runs/e312_p_s*/ends_card*.json`); deliver the probability (and a sigma, E318) | the metric is bound to a threshold; E318 lifted (D96) |
| S | S | frozen sensor features (D50) | pyramid input | no loss reaches camera or radar | decision of Mirko |
**Douglas-Peucker is NOT a step of the delivered graph:** the output stage `simplify_dp` is off by default (`PROBLEM.md` l.18; no call in the read-out path); the 0.3 m Douglas-Peucker is the metric's own vertex basis (`bootstrap.py:30-48`, "reported, not judged"). The control delivers 19.0 - 19.9 vertices per 100 m on that basis, above the budget of 14.3, so "within the budget" can only mean "control + 1.0" until a keep gate exists. **The plan's metric is the number of R steps: 8 at E324 s; the target of Q5 is at most 2 (prediction -6, hit window +-1).**

## 1b. What was measured before and must be read first
- **E273 (section 2.8, l.44 - 45, 68):** at 1 m, 12.1 % of the lane cells hold two or more pieces and 1.9 % crossing pieces (18.4 % / 3.7 % at 2 m), almost all of it connectors; connectors are 54 % of the pieces in the window, median length 19.8 m, 47 % longer than 20 m, and 31 % of them collide mid-length with another piece (lane pieces: 0 %). That is why B's rule is "connectors carry no nodes" (`qtargets.py:3-17`, intersection suppression) and why B has long links (`qlinks.py:104-106`). A read-only measurement of the GT connectors on every third VAL frame (275 junction frames, 3 701 connectors; the plan's critic, 2026-11-14): arc length median 19.0 m, p90 41.6 m, max 64 m; 73.5 % longer than a 12 m slot radius, 48 % longer than 20 m, 28 % longer than 30 m; 3.5 % of the 1 m cells on connectors are hit by two or more connectors with headings more than 45 degrees apart; node chains through junctions would add about 300 nodes per junction frame at 1 m (about 150 at 2 m) against 159 nodes per frame for all lanes in B.
- **E271 / E273 l.266:** the pyramid already carries a dense 1 m `centre1` head (existence, heading, sub-cell offset, a keep gate under a vertex budget, six event heats: `pyramid.py:4`, `qformat.py:13`); its read-out node F1 was 0.612 against 0.772 / 0.776 for B on the cell grid. That is why B uses 2 m cells. B's node F1 is 0.8525 / 0.8505 (`runs/e273/e273_p_s*/card.json`).
- **E294 (dense nodes), E295 (a position spread per node), E297 (soft keep gate):** plan cards, never run; E294 predicts a tie by itself and has a kill when the nodes are unscorable (a node F1 with greedy 2 m matching is not comparable between 1 m nodes and 2 m GT nodes).
- **One-to-one against duplicates:** E277 (soft close-node step) was cut on seed 0; E292 arm a (a trained Matrix-NMS layer) gave topo F1 -0.0265 / -0.0231, junction F1 -0.092 / -0.082, vertices +4.2 against c2 (E292 l.99 - 100), the cause named there: 32 % of the GT nodes lie within 1.75 m of each other; on a 1 m GT neighbouring nodes are 1 m apart.

""")

# ---- section 2 and 3
rep("**B** replaces the chain by one query head on the 1 m grid, which removes steps 1, 3, 4 and 5 by construction instead of imitating them.", "**B** replaces the chain by one query head on the 1 m grid, which would remove R1, R2, R5 and R6 by construction instead of imitating them, if Q0 shows that the representation carries the connectors (section 4).")
rep("| A1 | E325 learned wiring head; then the verdict at the equal-accuracy bar (snapped topo F1 >= E324 s - 0.005 on both seeds) | 1 (the wiring) | tonight; result section follows |", "| A1 | E325 learned wiring head; then the verdict at the equal-accuracy bar (snapped topo F1 >= E324 s - 0.005 on both seeds) | R1: the nearest-piece rule becomes a learned head argmax (a learned selection, not yet probabilities) | tonight; result section follows |")
rep("| 2 | card next, then E318 on the weekend queue |", "| T (reporting, not a rule step) | card next, then E318 on the weekend queue |")
rep("| feed 1 and 3 (better ends) |", "| feed R1 and R5 (better ends) |")
rep("| A4 | E297 soft chord keep gate: run it as written (plan card since 2026-11-11), judged by the vertex budget and topo F1 | 5 | card to be updated to the E324 s control, then one GPU job |", "| A4 | E297 soft chord keep gate: run it as written (plan card since 2026-11-11), judged by the vertex count against the control + 1.0 and topo F1 | none of R1 - R8 (makes the vertex count a trained quantity; Douglas-Peucker is not a delivered step) | card to be updated to the E324 s control, then one GPU job |")
rep("| A5 | the close-node step: no new experiment until track B decides; the one-to-one loss of Q-Lane removes the need (section 4) | 4 | parked, with the reason |", "| A5 | the close-node step: no new experiment until track B decides; the one-to-one loss of Q-Lane is a hypothesis against two negative results (section 1b) | R2 | parked, with the reason |")
open(p, "w", encoding="utf-8").write(s)
print("part 1 ok", len(s))
