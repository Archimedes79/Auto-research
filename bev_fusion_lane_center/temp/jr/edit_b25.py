"""Scratch (run once): the champion changed at 06:15 (E373): the same token head with none weight 0.25, decode link 0.2 / keep 0.6 / none gate off; the first recipe (none weight 1.0) is the runner-up."""
import json, os
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


def ls(t, head, new):
    L = t.split("\n")
    idx = [i for i, l in enumerate(L) if l.startswith(head)]
    assert len(idx) == 1, (head, idx)
    L[idx[0]] = new
    return "\n".join(L)


# ---------------- 01_intro
t = load("01_intro.md")
a = t.index("Two heads trained afterwards with fresh seeds (E371)")
b = t.index("\n", a)
t = t[:a] + ("Two heads trained afterwards with fresh seeds (E371) land within 0.004 (0.6618 / 0.6733). **One loss weight then changed**: the none target and the none class are weighted 0.25 instead of 1.0; on two fresh head seeds this reaches **0.6698 / 0.6769** (E373) and finds clearly more lanes (0.727 against 0.68), and by a rule written before the data it **replaces the first recipe as the champion** (D115 in the lane lead's record). The whole path takes 16.6 - 16.8 ms against 19.7 ms for the previous system. The champion is labelled *staged, lane head B' frozen* and is stage S1 of the design; the first recipe is the confirmed runner-up (a tie inside the band).") + t[b:]
t = rep(t, "(precision 0.69, more vertices, the junction F1 0.21 still below the plain head's 0.27)", "(precision 0.68, more vertices, the junction F1 0.20 still below the plain head's 0.27)")
t = rep(t, "and the challenger E373 is under confirmation (PENDING).", "and VAL has been read by many arms (the confirmation removes the choice of the head instance, not the reuse of VAL).")
save("01_intro.md", t)

# ---------------- 02_system
t = load("02_system.md")
t = rep(t, "The decode of record of stage S1 is keep 0.5, link 0.3 and the none gate on; the **decode of the champion candidate is keep 0.6, link 0.1, none gate off**, chosen on a TRAIN sample of 291 frames before the single VAL score (E366);",
        "The decode of record of stage S1 is keep 0.5, link 0.3 and the none gate on; the **decode of the champion is keep 0.6, link 0.2, none gate off**, chosen by a written rule on the head's own TRAIN sample of 291 frames before the single VAL score (the first recipe's cell was link 0.1, keep 0.6);")
t = rep(t, "the none slot when there is none), plus 0.5 x node-type cross-entropy", "the none slot when there is none), with the none target and the none class weighted **0.25 in the champion** (1.0 in the first recipe, the runner-up), plus 0.5 x node-type cross-entropy")
t = rep(t, "the whole path takes a median of 16.83 / 16.57 ms (B' alone 11.41 / 11.37 ms, the head alone 6.70 / 6.64 ms with 0.86 GMACs)", "the whole path takes a median of 16.83 / 16.57 ms (B' alone 11.41 / 11.37 ms, the head alone 6.70 / 6.64 ms with 0.86 GMACs; measured on the head of the first recipe, the champion has the identical architecture)")
save("02_system.md", t)

# ---------------- 04_experiments
t = load("04_experiments.md")
t = ls(t, "| E367 |", "| E367 | none-slot reweighting arms of the head (b_bal5, b_bal25), each read at its own TRAIN-chosen decode | b_bal5 0.6691 / 0.6696; b_bal25 0.6731 / 0.6728 | both PASS the quality gate; a three-way tie with the first recipe inside the band; b_bal25 named as the arm to confirm; the wide and long arms have not run |")
t = ls(t, "| **E371** |", "| E371 | the first recipe (none weight 1.0) retrained with fresh head seeds 2 and 3 on the two token caches, read at the frozen decode | 0.6618 / 0.6733 (F1@1.0 0.8402 / 0.8398, lanes found 0.685 / 0.685) | CONFIRMED; with the timing (16.8 / 16.6 ms) the first champion (D114), now the runner-up |")
t = ls(t, "| E373 |", "| **E373** | the none-weight-0.25 arm (b_bal25) on two fresh head seeds, read at the record decode and at the recipe's TRAIN-chosen cell | **0.6698 / 0.6769** at the cell link 0.2 / keep 0.6 / gate off (F1@1.0 0.8456 / 0.8461, lanes found 0.7274 / 0.7267); at the record decode 0.6424 / 0.6581 | CONFIRMED and **REPLACES** the first recipe by the pre-registered tie: **the champion** (the record decode misses F1@1.0 on cache 0 by 0.002) |")
save("04_experiments.md", t)

# ---------------- 05_quant
t = load("05_quant.md")
t = ls(t, "**In five lines", "**In five lines.** (1) The ladder of end-to-end systems of this project: plain lane head B 0.626 / 0.625, E2E with rule wiring 0.6912 / 0.6948, learned wiring 0.6931 / 0.6964, stack with the soft layer 0.6820 / 0.6854 (topo F1, VAL). (2) **The champion**, the token head on the predicted nodes of the frozen lane head, trained with the none target and the none class at weight 0.25 and read at a decode chosen on its TRAIN sample (link 0.2, keep 0.6, none gate off): on two fresh head seeds topo F1 **0.6698 / 0.6769**, F1@1.0 0.846 / 0.846, lanes found 0.727 / 0.727, +0.044 / +0.052 over the plain head with intervals excluding 0 (E373). The first recipe (none weight 1.0, link 0.1) is the confirmed runner-up: 0.6617 / 0.6767 and 0.6618 / 0.6733 (E366, E371), a tie inside the band with fewer lanes found (0.68). (3) It is cheaper than the previous system: the whole path takes 16.6 - 16.8 ms against 19.7 ms and 2.323 M parameters against 2.741 M, with one network stage and none of the five hard steps. (4) Plainly: precision 0.68, 27.4 - 27.8 vertices per 100 m (plain head 19), duplicates 0.058 / 0.061 (plain head 0.235) and a **junction F1 of 0.20, below the plain head's 0.27**: topology up, junction detection down. (5) It is stage S1 with the lane head frozen; PENDING: S2, S3, S4 and TEST (none exists).")
t = ls(t, "| 5 | E366:", "| 5 | E366: the first recipe (none weight 1.0), the decode chosen on the TRAIN sample, applied to VAL once | 0.6617 / 0.6767 | top-K 512 (budget), allowed-pair gate, output thresholds | E366 card, section 3 |")
t = ls(t, "| 6 | E371:", "| 6 | E371: the first recipe retrained with fresh head seeds 2 / 3, the decode frozen | 0.6618 / 0.6733 | as step 5 | E371 card, section 3 |")
t = ls(t, "| 7 | **the champion", "| 7 | **the champion**: the same head with the none weight 0.25 on fresh head seeds 2 / 3, its own TRAIN-chosen decode (E373) | 0.6698 / 0.6769 | as step 5, other threshold values | E373 card, section 3 |")
t = rep(t, "## B2. The same heads at the decode chosen on the TRAIN sample (E366): the champion candidate", "## B2. The champion and its runner-up: the token head at a decode chosen on the TRAIN sample (E366, E371, E373)")
t = rep(t, "The decode (link threshold 0.1, keep threshold 0.6, no none gate) was chosen by a written rule on the 291-frame TRAIN sample of the same heads, **before any VAL score of a variant**, and applied once to VAL. It is a change of the thresholds on the output only: the forward pass and the weights are those of E365.",
        "For each head a decode (link threshold, keep threshold, none gate) was chosen by a written rule on the 291-frame TRAIN sample of that head, **before any VAL score of a variant**, and applied once to VAL: the first recipe at link 0.1 / keep 0.6 / gate off, the champion at link 0.2 / keep 0.6 / gate off. A decode is a change of thresholds on the output only; the forward pass is that of the trained head.")
t = ls(t, "| **S1, chosen decode (E366), seed 1**", "| **S1 first recipe, chosen decode (E366), seed 1** | **0.6767** | 0.687 / 0.666 | **0.8422** | **0.6905** | 0.204 | 0.054 | 25.7 |\n| S1 first recipe, fresh heads (E371), seed 0 / 1 | 0.6618 / 0.6733 | 0.684 / 0.641 and 0.683 / 0.664 | 0.8402 / 0.8398 | 0.6849 / 0.6846 | 0.205 / 0.213 | 0.059 / 0.063 | 25.3 / 25.2 |\n| none weight 0.25 (E367 b_bal25, selection heads), seed 0 / 1 | 0.6731 / 0.6728 | | 0.849 / 0.845 | 0.741 / 0.725 | 0.194 / 0.197 | 0.056 / 0.061 | 29.1 / 27.9 |\n| **CHAMPION: none weight 0.25, fresh heads (E373), cell 0.2 / 0.6 / off, seed 0** | **0.6698** | 0.680 / 0.660 | **0.8456** | **0.7274** | 0.203 | 0.058 | 27.4 |\n| **CHAMPION (E373), seed 1** | **0.6769** | 0.681 / 0.673 | **0.8461** | **0.7267** | 0.197 | 0.061 | 27.8 |\n| champion at the record decode (0.3 / 0.5 / gate on), seed 0 / 1 | 0.6424 / 0.6581 | 0.776 / 0.548 and 0.780 / 0.569 | 0.8162 / 0.8203 | 0.641 / 0.648 | 0.248 / 0.241 | 0.017 / 0.017 | 22.4 / 22.9 |")
t = rep(t, "**Caveats, as the card states them.**", "**The champion (E373).** The same head trained with the none target and the none class at weight 0.25 (one loss weight, nothing added), on two fresh head seeds, read at the cell its recipe chooses on the TRAIN sample (link 0.2, keep 0.6, none gate off): topo F1 0.6698 / 0.6769, F1@1.0 0.8456 / 0.8461, lanes found 0.7274 / 0.7267; against plain B +0.044 [+0.028, +0.059] and +0.052 [+0.034, +0.068]. At the record decode (no tuned thresholds) it misses the gate on cache 0 by 0.002 in F1@1.0 (0.8162 against 0.8182) and is the cleanest operating point of the night (junction F1 0.24 - 0.25, duplicates 0.017, 22.4 - 22.9 vertices). Against the first recipe the champion has the same topo F1 inside the band (+0.008 / +0.002), a slightly higher F1@1.0 (+0.008 / +0.005) and clearly more lanes found (+0.048 / +0.039), at slightly more vertices (27.4 - 27.8 against 25.2 - 25.5), the same duplicates and junction F1; the rule written before the data (a quality tie with the first champion on both caches) therefore **replaces** the first recipe. Said plainly: b_bal25 was named among the arms read on VAL; VAL has been read by every arm of the night; the confirmation removes the selection of the head instance, not the reuse of VAL; the seed-twin interval and the frame-order check of the card are not done yet.\n\n**Caveats, as the card states them.**")
t = rep(t, "the champion is declared by the written rule after the confirmation on two fresh head seeds (E371) and the timing (section D); it is **staged, lane head B' frozen**.", "the champion is declared by the written rule after the confirmation on two fresh head seeds (E373) and the timing (section D, measured on the first recipe's head, identical architecture); it is **staged, lane head B' frozen**.")
t = ls(t, "| **champion, whole path**", "| **champion, whole path** | **2.323** | **16.83 / 16.57** (p90 19.09 / 20.39), 16.6 - 16.8 ms in the quiet window | **2.9 - 3.1 ms and 0.42 M parameters below E324 s**; measured on the head of the first recipe, the champion has the identical architecture (a loss weight differs); the tool flagged the window as not quiet (GPU up to 10 % before the run): contention can only slow the path, so the comparison is conservative; the tail (p90) is not better than E324 s's median |")
t = rep(t, "the junction F1 of the champion (0.20 - 0.21) is below", "the junction F1 of the champion (0.20) is below")
t = rep(t, "precision is 0.69. PENDING:", "precision is 0.68. PENDING:")
save("05_quant.md", t)

# ---------------- 07_winners
t = load("07_winners.md")
t = ls(t, "| | E324 s, E2E read-out", "| | E324 s, E2E read-out (reference) | token champion: the head with none weight 0.25 at its TRAIN-chosen decode (E373) |")
t = ls(t, "| topo F1 |", "| topo F1 | 0.6912 / 0.6948 | 0.6698 / 0.6769 (fresh heads, E373); selection heads 0.6731 / 0.6728 (E367); floors 0.6359 / 0.6349, plain head 0.6259 / 0.6249; runner-up (none weight 1.0): 0.6617 / 0.6767, 0.6618 / 0.6733 |")
t = ls(t, "| F1@1.0, lanes found |", "| F1@1.0, lanes found | 0.8777 / 0.8786, 0.7653 / 0.7648 | 0.8456 / 0.8461, 0.7274 / 0.7267 (runner-up 0.836 - 0.842, 0.675 - 0.691) |")
t = ls(t, "| time per frame", "| time per frame (whole path, quiet window, 24 frames, batch 1, bf16) | 19.7 ms | 16.6 - 16.8 ms (identical architecture to the runner-up, which was timed) |")
t = ls(t, "| code in git", "| code in git | `training/lanegraph/` (the package of 2026-11-16, no weights); `winners/e324s_lanegraph/` | `winners/e365_token_champion/` (the head, the loss, the decode, the loader, the cache, the trainer, the evaluation and their tests; no weights; it needs the repository; the champion differs from the runner-up by the option `--none-weight 0.25` and the decode values) |")
t = ls(t, "| weights and run folder", "| weights and run folder | `runs/e324_s_s0`, `runs/e324_s_s1` | champion heads `runs/e373_cb_s0` (cache 0, head seed 2) and `runs/e373_cb_s1` (cache 1, head seed 3), selection heads `runs/e367_b25_s0 / s1`; runner-up `runs/e365_s1_s0 / s1`, `runs/e371_cf_s0 / s1`; frozen lane head B' `runs/e292_a_s0 / s1`; token cache `runs/e365_cache/be292_a_{0,1}_K512_9ad1d4ce30_v1` (derived); none in git |")
a = t.index("**A challenger is under confirmation.**")
t = t[:a] + "**The champion changed during the night.** The first champion (D114) was the head with none weight 1.0 at link 0.1 (topo F1 0.6617 / 0.6767 and 0.6618 / 0.6733). The arm with none weight 0.25 (E367) tied with it inside the band, and its confirmation on two fresh head seeds (E373) satisfied the rule written before the data (a quality tie on both caches), so it replaced the first recipe; the first recipe is the confirmed runner-up and both are in the same winner folder (one option, `--none-weight`, and the decode values differ).\n"
save("07_winners.md", t)

# ---------------- 08_outlook
t = load("08_outlook.md")
t = rep(t, "but passes its quality half at a decode chosen on the TRAIN sample (0.6617 / 0.6767, +0.036 / +0.052 over the plain head): the head ranks links well, the operating point was the problem. It costs precision (0.69), vertices and junction F1.",
        "but passes its quality half at a decode chosen on the TRAIN sample (0.6617 / 0.6767, +0.036 / +0.052 over the plain head): the head ranks links well, the operating point was the problem. One loss weight (the none target at 0.25) then gives the champion, 0.6698 / 0.6769 on fresh seeds with more lanes found (0.727). It costs precision (0.68), vertices (27.4 - 27.8 per 100 m) and junction F1 (0.20).")
t = rep(t, "the junction F1 of the champion (0.20 - 0.21) is below", "the junction F1 of the champion (0.20) is below")
t = rep(t, "**Next steps, very brief.** The challenger E373 (the none-slot arm on fresh seeds); S2,", "**Next steps, very brief.** The seed-twin interval and the frame-order check of E373 (CPU, open); S2,")
save("08_outlook.md", t)

# ---------------- A1 repeat
t = load("A1_repeat.md")
t = rep(t, "`bash tools/env/_py14.sh jtoken_train.py --name e365_s1_s<S> --seed S --ident e292_a --steps 12000 --bs 8 --lr 1e-3 --workers 4` (AdamW,", "`bash tools/env/_py14.sh jtoken_train.py --name e373_cb_s<S> --seed S --ident e292_a --none-weight 0.25 --steps 12000 --bs 8 --lr 1e-3 --workers 4` (the champion; the first recipe is the same command with the default none weight 1.0; AdamW,")
t = rep(t, "none weight 1.0, plus 0.5 x node-type", "none weight 0.25 (champion) or 1.0 (first recipe), plus 0.5 x node-type")
t = rep(t, "`jtoken_eval.py dump --run e365_s1_s<S> --seed S`, then `jtoken_eval.py score --run e365_s1_s<S> --seed S --link-thr 0.1 --keep-thr 0.6 --no-none-gate`: the decode chosen on the 291-frame TRAIN sample by the written rule of E366", "`jtoken_eval.py dump --run e373_cb_s<S> --seed S`, then `jtoken_eval.py score --run e373_cb_s<S> --seed S --link-thr 0.2 --keep-thr 0.6 --no-none-gate` (the first recipe: link 0.1): the decode chosen on the 291-frame TRAIN sample by the written rule of E366")
t = rep(t, "the head at the decode of record: 0.5486 / 0.5743 (E365); at the chosen decode: 0.6617 / 0.6767 (E366); fresh head seeds 2 / 3: 0.6618 / 0.6733 (E371);", "the first recipe at the decode of record: 0.5486 / 0.5743 (E365); at its chosen decode: 0.6617 / 0.6767 (E366), fresh head seeds 2 / 3: 0.6618 / 0.6733 (E371); **the champion (none weight 0.25, fresh seeds 2 / 3, cell 0.2 / 0.6 / off): 0.6698 / 0.6769 (E373)**;")
t = ls(t, "| E371 |", "| E371 | the first recipe, fresh head seeds 2 / 3 at the frozen decode | CONFIRMED: 0.6618 / 0.6733 | `E371_e366-confirmation-fresh-head-seeds` |\n| E373 | the none-weight-0.25 arm on fresh head seeds 2 / 3 | CONFIRMED, replaces the first recipe: 0.6698 / 0.6769 | `E373_b-bal5-confirmation-fresh-head-seeds` (the folder keeps its v1 name; the card is for b_bal25) |")
save("A1_repeat.md", t)

# ---------------- ladder + report.json
lj = os.path.join(P, "figures", "ladder.json")
s = json.load(open(lj, encoding="utf-8"))
rows = [r for r in s["rows"] if "challenger" not in r["label"]]
for r in rows:
    if r["label"].startswith("token champion: S1 head"):
        r["label"] = "token head, first recipe (E366)"
    if r["label"].startswith("token champion, fresh head seeds"):
        r["label"] = "token head, first recipe, fresh seeds (E371)"
rows.append({"label": "token champion: none weight 0.25 (E373)", "values": [0.6698, 0.6769]})
s["rows"] = rows
json.dump(s, open(lj, "w", encoding="utf-8"), indent=1)
rj = os.path.join(RP, "report.json")
r = json.load(open(rj, encoding="utf-8"))
r["abstract"] = ("A lane-centre graph (centre lines, forks, merges and junction connectors) is computed from camera and radar features. The earlier system was made end to end "
                 "differentiable step by step and reaches topo F1 0.6912 / 0.6948 on VAL (plain lane head 0.6259 / 0.6249) but keeps five hard steps. We replace its junction machinery "
                 "by one token head: the predicted nodes of the frozen lane head are tokens, a none slot prunes softly and the per-source softmax of the last self-attention layer is the wiring. "
                 "With the none target weighted 0.25 and a decode chosen on the TRAIN sample it reaches 0.6698 / 0.6769 on two fresh seeds, +0.044 / +0.052 over the plain head, in 16.6 - 16.8 ms against 19.7 ms "
                 "and with 0.127 M parameters on top of the frozen lane head, with none of the five hard steps. It pays plainly: precision 0.68, 27 vertices per 100 m and a junction F1 of 0.20, "
                 "below the plain head's 0.27. It is stage S1 with the lane head frozen; the past, the map and a joint fine-tune are not in it, VAL has been read by every arm, and no TEST number exists.")
json.dump(r, open(rj, "w", encoding="utf-8"), indent=1)
print("ok")
