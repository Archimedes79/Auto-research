"""Scratch (run once): the champion CANDIDATE (E365 S1 heads at the E366 decode) into the paper report draft (results/E2E_Champion_Report/parts)."""
import json, os, re
R = os.environ["BFLC_ROOT"]
RP = os.path.join(R, "results", "E2E_Champion_Report")
P = os.path.join(RP, "parts")


def load(fn):
    return open(os.path.join(P, fn), encoding="utf-8").read()


def save(fn, t):
    open(os.path.join(P, fn), "w", encoding="utf-8").write(t)


def rep(t, old, new):
    assert t.count(old) == 1, (t.count(old), old[:90])
    return t.replace(old, new)


def line_starting(t, head, new):
    L = t.split("\n")
    idx = [i for i, l in enumerate(L) if l.startswith(head)]
    assert len(idx) == 1, (head, idx)
    L[idx[0]] = new
    return "\n".join(L)


CAND = "the E365 S1 head (0.127 M parameters on the frozen lane head B') read at the decode chosen on the TRAIN sample (E366)"

# ---- 01_intro
t = load("01_intro.md")
t = rep(t, "Its first stage, S1, **missed** the written gate: 0.5486 / 0.5743 against floors of 0.6359 / 0.6349. PENDING: the later arms and the champion.",
        "Its first stage, S1, **missed** the written gate at the decode of record (0.5486 / 0.5743 against floors of 0.6359 / 0.6349), but the head ranks the links well: read at a decode chosen on the TRAIN sample (link threshold 0.1, keep threshold 0.6, no none gate; chosen before the single VAL score) the same heads reach **0.6617 / 0.6767** and **pass the quality half of the gate on both seeds** (E366). That is the **champion candidate**; it is not final until the quiet-host timing and the confirmation on two fresh head seeds are through (PENDING).")
t = rep(t, "4. What S1 taught us: the head is precise (0.86) and conservative (recall 0.40 - 0.43), the junction links are the weak part, and a one-line rule on the same tokens is better than the head (0.6265 / 0.6281) (chapters 5, 6).",
        "4. What S1 taught us: at the decode of record the head is precise (0.86) and conservative (recall 0.40 - 0.43), and a one-line rule on the same tokens is better than it (0.6265 / 0.6281); at the looser decode the recall is recovered at a price (precision 0.69, more vertices, the junction F1 0.21 still below the plain head's 0.27): the junction links remain the weak part (chapters 5, 6).")
t = rep(t, "**Status.** PENDING (the champion).", "**Status.** A champion candidate exists (above); PENDING: the quiet-host timing of the whole path, the confirmation on two fresh seeds, and the stages S2 (past), S3 (map) and S4 (joint fine-tune).")
save("01_intro.md", t)

# ---- 02_system: the decode
t = load("02_system.md")
t = rep(t, "on the output only: a node is kept when p_none < 0.5, a link when p >= 0.3 and p >= p(none) and both ends are kept;",
        "on the output only (the decode): a node is kept when p_none is below the keep threshold, a link when p is above the link threshold (and, optionally, above p(none)) and both ends are kept. The decode of record of stage S1 is keep 0.5, link 0.3 and the none gate on; the **decode of the champion candidate is keep 0.6, link 0.1, none gate off**, chosen on a TRAIN sample of 291 frames before the single VAL score (E366);")
save("02_system.md", t)

# ---- 04_experiments
t = load("04_experiments.md")
t = line_starting(t, "| **E365** |", "| **E365** | stage S1 on the predicted nodes of the frozen lane head: soft pruning, self-attention wiring, node-type softmax | 0.5486 / 0.5743 at the decode of record | MISS at that decode (written gate) |")
t = line_starting(t, "| E366 |", "| **E366** | the same heads, the decode (link threshold, keep threshold, none gate) chosen on the TRAIN sample, applied to VAL once | **0.6617 / 0.6767** (F1@1.0 0.836 / 0.842, lanes found 0.675 / 0.691) | PASS-DECODE: the quality half of the gate holds on both seeds; champion candidate |")
save("04_experiments.md", t)

# ---- 05_quant
t = load("05_quant.md")
t = line_starting(t, "**In five lines", "**In five lines.** (1) The ladder of end-to-end systems of this project: plain lane head B 0.626 / 0.625, E2E with rule wiring 0.6912 / 0.6948, learned wiring 0.6931 / 0.6964, stack with the soft layer 0.6820 / 0.6854 (topo F1, VAL). (2) The token head, first stage S1 (E365, the predicted nodes of the frozen B'): 0.5486 / 0.5743 at the decode of record, a MISS by the written gate. (3) The same heads at a decode chosen on the TRAIN sample (E366): **0.6617 / 0.6767**, F1@1.0 0.836 / 0.842, lanes found 0.675 / 0.691: the quality half of the gate holds on both seeds (+0.036 / +0.052 over the plain head, intervals excluding 0); this is the **champion candidate**, 2.32 M parameters against 2.741 M. (4) The price: precision 0.86 to 0.69, 25 vertices per 100 m (plain head 19), junction F1 0.209 / 0.204 against 0.272 for the plain head. (5) PENDING: the ms of the whole path on a quiet host, the confirmation on two fresh head seeds, and the stages S2, S3, S4.")
t = line_starting(t, "**Verdict by the written rule: MISS**", t.split("\n")[[l.startswith("**Verdict by the written rule: MISS**") for l in t.split("\n")].index(True)].replace("**Verdict by the written rule: MISS**", "**Verdict by the written rule at the decode of record: MISS**", 1))
sec = """## B2. The same heads at the decode chosen on the TRAIN sample (E366): the champion candidate
The decode (link threshold 0.1, keep threshold 0.6, no none gate) was chosen by a written rule on the 291-frame TRAIN sample of the same heads, **before any VAL score of a variant**, and applied once to VAL. It is a change of the thresholds on the output only: the forward pass and the weights are those of E365.

Table: Stage S1 at the decode of record and at the chosen decode, against plain B and its controls (VAL 873 frames, seed 0 / seed 1).

| VAL 873, delivered graph | topo F1 | topo P / R | F1@1.0 | lanes found | junction F1 | dup_frac | vertices / 100 m |
|---|---|---|---|---|---|---|---|
| floors of the gate | >= 0.6359 / 0.6349 | | >= 0.8182 / 0.8060 | >= 0.6066 / 0.5937 | | | |
| plain B (stored) | 0.6259 / 0.6249 | 0.670 / 0.587 (s0) | 0.8282 / 0.8160 | 0.6266 / 0.6137 | 0.272 (s0) | 0.235 (s0) | 19.29 (s0) |
| one-line rule on the same tokens | 0.6265 / 0.6281 | | 0.773 / 0.773 | 0.586 / 0.586 | 0.163 | 0.005 / 0.004 | |
| S1, decode of record | 0.5486 / 0.5743 | 0.858 / 0.403 and 0.853 / 0.433 | 0.752 / 0.763 | 0.500 / 0.525 | 0.177 / 0.197 | 0.004 | 18.5 / 18.6 |
| **S1, chosen decode (E366), seed 0** | **0.6617** | 0.694 / 0.633 | **0.8359** | **0.6747** | 0.209 | 0.055 | 25.2 |
| **S1, chosen decode (E366), seed 1** | **0.6767** | 0.687 / 0.666 | **0.8422** | **0.6905** | 0.204 | 0.054 | 25.7 |

**Against the gate:** topo F1 0.6617 / 0.6767 against the floors 0.6359 / 0.6349 holds on both seeds (+0.026 / +0.042 over the floor); F1@1.0 0.836 / 0.842 against 0.818 / 0.806 and lanes found 0.675 / 0.691 against 0.607 / 0.594 hold; the parameters of the whole path (2.32 M against 2.741 M) hold; the milliseconds are open (the only measurement, 19.0 ms against 19.7 ms, was taken on a loaded host and is not usable); one training stage and no rule step in the delivered graph: the simplicity condition holds. **Paired scene bootstrap (topo F1, 1000 draws):** against plain B +0.036 [+0.018, +0.052] and +0.052 [+0.032, +0.070]; against the decode of record +0.113 [+0.090, +0.140] and +0.102 [+0.081, +0.129]; every interval excludes 0.

**What it costs, honestly.** Precision falls from 0.86 to 0.69; the duplicate share rises from 0.004 to 0.055 (still far below the plain head's 0.235); the vertex density rises (25 against 19 per 100 m); the junction F1 (0.209 / 0.204) stays below the plain head's 0.272, so the junction links remain the weak part; on the TRAIN sample the same decode scores 0.720 / 0.725, i.e. the TRAIN-to-VAL drop is 0.06 / 0.05.

**Caveats, as the card states them.** The decode was selected on the TRAIN sample of the same heads (the heads were trained on those frames) and is one of 24 cells; the paired intervals use plain B's stored per-frame records; the cache-against-live parity check of E365 failed its tolerance on 0.1 % of the nodes; **a champion is not declared until the recipe is confirmed on two fresh head seeds and the whole path is timed on a quiet host (PENDING)**.

"""
t = rep(t, "## C. The oracle-node stage of the same design", sec + "## C. The oracle-node stage of the same design")
save("05_quant.md", t)

# ---- 07_winners / 08_outlook
t = load("07_winners.md")
t = rep(t, "| | E324 s, E2E read-out (reference) | token champion |", "| | E324 s, E2E read-out (reference) | token champion candidate (E365 S1 heads, E366 decode) |")
t = rep(t, "| topo F1 | 0.6912 / 0.6948 | PENDING |", "| topo F1 | 0.6912 / 0.6948 | 0.6617 / 0.6767 (the quality half of the gate holds; confirmation on two fresh seeds PENDING) |")
t = rep(t, "| F1@1.0, lanes found | 0.8777 / 0.8786, 0.7653 / 0.7648 | PENDING |", "| F1@1.0, lanes found | 0.8777 / 0.8786, 0.7653 / 0.7648 | 0.836 / 0.842, 0.675 / 0.691 |")
t = rep(t, "| parameters | 2.741 M | PENDING (S1: 2.32 M, of which the head 0.127 M) |", "| parameters | 2.741 M | 2.32 M (the frozen lane head B' 2.196 M + the head 0.127 M) |")
t = rep(t, "| training stages | three, plus the attachment | PENDING (S1: one) |", "| training stages | three, plus the attachment | one (the head; B' is frozen) |")
t = rep(t, "| hard steps left | keypoint peaks + top-K 32, wiring rule (or a learned argmax), close-node step, thresholds, candidate gates | PENDING (S1: top-K 512 budget, the allowed-pair gate, output thresholds) |",
        "| hard steps left | keypoint peaks + top-K 32, wiring rule (or a learned argmax), close-node step, thresholds, candidate gates | the top-K 512 budget, the allowed-pair gate, the output thresholds |")
save("07_winners.md", t)

t = load("08_outlook.md")
t = rep(t, "- On the predicted nodes of the frozen lane head the first stage S1 does not reach the written level (0.5486 / 0.5743); the head is below a one-line rule on the same tokens.",
        "- On the predicted nodes of the frozen lane head the first stage S1 misses the written level at its decode of record (0.5486 / 0.5743) but passes its quality half at a decode chosen on the TRAIN sample (0.6617 / 0.6767, +0.036 / +0.052 over the plain head): the head ranks links well, the operating point was the problem. It costs precision (0.69), vertices and junction F1.")
save("08_outlook.md", t)

# ---- ladder + report.json
lj = os.path.join(P, "figures", "ladder.json")
s = json.load(open(lj, encoding="utf-8"))
rows = s["rows"]
rows[4]["label"] = "token head S1, decode of record (E365)"
rows.insert(5, {"label": "token head S1, chosen decode (E366)", "values": [0.6617, 0.6767]})
rows[6]["label"] = "later arms / confirmation (PENDING)"
json.dump(s, open(lj, "w", encoding="utf-8"), indent=1)
rj = os.path.join(RP, "report.json")
r = json.load(open(rj, encoding="utf-8"))
r["abstract"] = ("A lane-centre graph (centre lines, forks, merges and junction connectors) is computed from camera and radar features. Step by step the earlier system was made "
                 "end to end differentiable and reaches topo F1 0.6912 / 0.6948 on VAL (plain lane head: 0.6259 / 0.6249) with five hard steps left. We then replace the junction "
                 "machinery by one token head: the predicted nodes of the frozen lane head are tokens, a none slot prunes softly and the per-source softmax of the last self-attention "
                 "layer is the wiring. Read at a decode chosen on the TRAIN sample it reaches 0.6617 / 0.6767, +0.036 / +0.052 over the plain head, with 2.32 M parameters against "
                 "2.741 M and none of the five hard steps; it pays with precision (0.69), more vertices and a junction F1 still below the plain head. "
                 "PENDING: the timing on a quiet host, the confirmation on two fresh seeds, and the stages that add the past and the map.")
json.dump(r, open(rj, "w", encoding="utf-8"), indent=1)
print("ok")
