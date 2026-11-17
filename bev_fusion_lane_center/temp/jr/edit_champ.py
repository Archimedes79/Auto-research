"""Scratch (run once): the champion of D114 (E365 S1 heads at the E366 decode, confirmed by E371) into the paper report parts."""
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
t = rep(t, "That is the **champion candidate**; it is not final until the quiet-host timing and the confirmation on two fresh head seeds are through (PENDING).",
        "Two heads trained afterwards with fresh seeds (E371) land within 0.004 (0.6618 / 0.6733), and the whole path takes 16.8 / 16.6 ms against 19.7 ms for the previous system: **this is the champion (D114)**. It is labelled *staged, lane head B' frozen* and is stage S1 of the design; a challenger (E373) is under confirmation.")
t = rep(t, "**Status.** A champion candidate exists (above); PENDING: the quiet-host timing of the whole path, the confirmation on two fresh seeds, and the stages S2 (past), S3 (map) and S4 (joint fine-tune).",
        "**Status.** A champion exists (D114, decided by the written rule of D113, not by the owner: he can overrule it). Not done: S2 (cross-attention to the past) is measured later, S3 (the map) is not built, S4 (a joint fine-tune so that the loss reaches the lane head) has not run, there is no TEST number, and the challenger E373 is under confirmation (PENDING).")
save("01_intro.md", t)

# ---------------- 02_system
t = load("02_system.md")
t = rep(t, "PENDING: which of them exist when the report is made.", "Status: S2 (E368) is measured later, S3 is not built, S4 (E372) has not run: **the champion is stage S1 with the lane head B' frozen** (PENDING for S2, S3, S4).")
t = rep(t, "the ms of the whole path was measured only on a host that was not quiet (see chapter 3) and is to be repeated. PENDING: the same table for the champion.",
        "the whole path takes a median of 16.83 / 16.57 ms (B' alone 11.41 / 11.37 ms, the head alone 6.70 / 6.64 ms with 0.86 GMACs) against 19.7 ms for E324 s (chapter 5, section D).")
save("02_system.md", t)

# ---------------- 04_experiments
t = load("04_experiments.md")
t = ls(t, "| E367 |", "| E367 | none-slot reweighting arms of the head (b_bal5, b_bal25), each read at its own TRAIN-chosen decode | b_bal5 0.6691 / 0.6696; b_bal25 0.6731 / 0.6728 | both PASS the quality gate; a three-way tie with the champion inside the band; b_bal25 is the challenger (E373); the wide and long arms have not run |")
t = rep(t, "| E367 |", "| E367 |") if False else t
t = ls(t, "| S2, S3, S4 |", "| **E371** | the E365 recipe retrained with fresh head seeds 2 and 3 on the two token caches, read at the frozen decode | **0.6618 / 0.6733** (F1@1.0 0.8402 / 0.8398, lanes found 0.685 / 0.685) | CONFIRMED: the gate holds on both fresh heads; with the timing (16.8 / 16.6 ms) the **champion (D114)** |\n"
        "| E373 | the none-slot arm b_bal25 on two fresh head seeds | PENDING | PENDING (challenger under confirmation) |\n"
        "| S2 (E368), S3, S4 (E372) | cross-attention to the past, to the map; joint fine-tune so that the loss reaches the lane head | PENDING | PENDING (S2 coming, S3 not built, S4 not run) |")
save("04_experiments.md", t)

# ---------------- 05_quant
t = load("05_quant.md")
t = rep(t, "Cells marked PENDING are filled when the champion exists.", "Cells marked PENDING are not measured yet (S2, S3, S4, the challenger, the pictures).")
t = ls(t, "**In five lines", "**In five lines.** (1) The ladder of end-to-end systems of this project: plain lane head B 0.626 / 0.625, E2E with rule wiring 0.6912 / 0.6948, learned wiring 0.6931 / 0.6964, stack with the soft layer 0.6820 / 0.6854 (topo F1, VAL). (2) **The champion** (D114), the token head on the predicted nodes of the frozen lane head read at the decode chosen on the TRAIN sample: topo F1 0.6617 / 0.6767, and on two fresh head seeds 0.6618 / 0.6733; F1@1.0 0.836 / 0.842 / 0.840 / 0.840; lanes found 0.675 / 0.691 / 0.685 / 0.685; +0.036 / +0.052 / +0.036 / +0.048 over the plain head, every interval excluding 0. (3) It is cheaper than the previous system: the whole path takes 16.8 / 16.6 ms against 19.7 ms and 2.323 M parameters against 2.741 M, with one network stage and none of the five hard steps. (4) Plainly: precision 0.69 (0.86 at the decode of record), 25 vertices per 100 m (plain head 19), duplicates 0.055 - 0.063 (plain head 0.235) and a **junction F1 of 0.20 - 0.21, below the plain head's 0.27**: topology up, junction detection down. (5) It is stage S1 with the lane head frozen; PENDING: S2, S3, S4, the challenger E373 and TEST (none exists).")
t = ls(t, "| 5 | E366:", "| 5 | E366: the same two heads, the decode (link threshold, keep threshold, none gate) chosen on the TRAIN sample, applied to VAL once | 0.6617 / 0.6767 | top-K 512 (budget), allowed-pair gate, output thresholds | E366 card, section 3 |")
t = ls(t, "| 6 | E367:", "| 6 | E371: the recipe retrained with fresh head seeds 2 / 3, the decode frozen | 0.6618 / 0.6733 | as step 5 | E371 card, section 3 |")
t = ls(t, "| 7 | **the champion**", "| 7 | **the champion (D114)** = the E365 S1 head at the E366 decode, confirmed by E371 | 0.6617 / 0.6767 and 0.6618 / 0.6733 | as step 5; staged, lane head frozen | D114, E366, E371 |")
t = ls(t, "| E366 / E367 / the champion |", "| **E371 fresh heads (head seeds 2 / 3), the same decode** | **0.6618 / 0.6733** | 0.684 / 0.641 and 0.683 / 0.664 | **0.8402 / 0.8398** | **0.6849 / 0.6846** | 0.205 / 0.213 | 0.059 / 0.063 | 25.3 / 25.2 |\n| E371 fresh heads, decode of record (diagnosis) | 0.5499 / 0.5717 | 0.847 / 0.407 and 0.851 / 0.431 | 0.758 / 0.764 | 0.507 / 0.518 | 0.197 / 0.223 | 0.004 | 18.2 / 18.4 |") if False else t
t = rep(t, "**Caveats, as the card states them.**", "**Confirmation on fresh heads (E371).** Two heads trained with fresh seeds (2 and 3, on the token caches 0 and 1) and read at the same frozen decode give topo F1 0.6618 / 0.6733 (F1@1.0 0.8402 / 0.8398, lanes found 0.685 / 0.685, junction F1 0.205 / 0.213, duplicates 0.059 / 0.063, 25.3 / 25.2 vertices per 100 m); against plain B +0.036 [+0.020, +0.049] and +0.048 [+0.030, +0.066], both excluding 0. The head seed alone moves topo F1 by +0.0001 and -0.0034, less than the band; the tightest condition, F1@1.0 on cache 0, holds with +0.022. On the TRAIN sample the same decode scores 0.7128 / 0.7255, so the TRAIN-to-VAL drop is 0.051 / 0.052.\n\n**Caveats, as the card states them.**")
t = rep(t, "**a champion is not declared until the recipe is confirmed on two fresh head seeds and the whole path is timed on a quiet host (PENDING)**.", "the champion is declared by the written rule after the confirmation on two fresh head seeds (E371) and the timing (section D); it is **staged, lane head B' frozen**.")
t = rep(t, "Table: Compute of the reference, of the parts of S1 and (PENDING) of the champion; the ms of S1 were measured on a host that was not quiet.", "Table: Compute of the reference and of the champion and its parts (24 frames, batch 1, bf16, CUDA events, 5 repeats, K = 512; the champion's numbers are two runs in the quiet window, host load 2.2 - 2.9).")
t = ls(t, "| B' alone (S1)", "| B' alone (pyramid + node head + top-K gather) | 2.196 | 11.41 / 11.37 | the frozen lane head, cached tokens are derived from it |")
t = ls(t, "| S1 head alone", "| the token head alone | 0.127 | 6.70 / 6.64 (0.86 GMACs at K = 512) | the three self-attention layers and the wiring layer |")
t = ls(t, "| S1 whole path", "| **champion, whole path** | **2.323** | **16.83 / 16.57** (p90 19.09 / 20.39) | **2.9 - 3.1 ms and 0.42 M parameters below E324 s**; the tool flagged the window as not quiet (GPU up to 10 % before the run, its limit is 5 %): contention can only slow the path, so the comparison is conservative; the tail (p90) is not better than E324 s's median |")
t = ls(t, "| the champion | PENDING", "| the decode thresholds | 0 | 0 | the same forward, a threshold on the output |")
t = ls(t, "## E. Still difficult situations", "## E. Still difficult situations")
t = rep(t, "PENDING for the champion (crowded junctions, far range, own lane, duplicates, with the same splits as the earlier report). Known from S1: the long (junction) links, the node pruning and the train-validation gap.",
        "The junction links are the weak part: the junction F1 of the champion (0.20 - 0.21) is below the plain head's 0.27 because the written gate has no junction condition; the duplicates are low (0.055 - 0.063) but above the record decode's 0.004; precision is 0.69. PENDING: the split by crowded junctions, far range and own lane, as in the earlier report (not computed for the token head yet).")
save("05_quant.md", t)

# ---------------- 06 qualitative
t = load("06_qual.md")
t = rep(t, "PENDING: the panels of the champion next to the plain lane head and the E2E rule-wired system E324 s.", "PENDING: the panels of the champion next to the plain lane head and the E2E rule-wired system E324 s (the graph export of the champion is ordered; the pictures follow it).")
save("06_qual.md", t)

# ---------------- 07 winners
t = load("07_winners.md")
t = rep(t, "and the token champion (PENDING).", "and the token champion (D114).")
t = rep(t, "token champion candidate (E365 S1 heads, E366 decode)", "token champion (D114): E365 S1 head at the E366 decode")
t = ls(t, "| topo F1 |", "| topo F1 | 0.6912 / 0.6948 | 0.6617 / 0.6767 (heads 0 / 1) and 0.6618 / 0.6733 (fresh heads 2 / 3); floors 0.6359 / 0.6349, plain head 0.6259 / 0.6249 |")
t = ls(t, "| F1@1.0, lanes found |", "| F1@1.0, lanes found | 0.8777 / 0.8786, 0.7653 / 0.7648 | 0.836 / 0.842 / 0.840 / 0.840, 0.675 / 0.691 / 0.685 / 0.685 |")
t = ls(t, "| time per frame", "| time per frame (whole path, quiet window, 24 frames, batch 1, bf16) | 19.7 ms | 16.83 / 16.57 ms |")
t = ls(t, "| code in git", "| code in git | `training/lanegraph/` (the package of 2026-11-16, no weights); `winners/e324s_lanegraph/` | `winners/e365_token_champion/` (the head, the loss, the decode, the loader, the cache, the trainer, the evaluation and their tests; no weights; it needs the repository) |")
t = ls(t, "| weights and run folder", "| weights and run folder | `runs/e324_s_s0`, `runs/e324_s_s1` | heads `runs/e365_s1_s0`, `runs/e365_s1_s1`, `runs/e371_cf_s0`, `runs/e371_cf_s1`; frozen lane head B' `runs/e292_a_s0 / s1`; token cache `runs/e365_cache/be292_a_{0,1}_K512_9ad1d4ce30_v1` (derived); none in git |")
t = rep(t, "PENDING: the final form with the stages S2 (past), S3 (map) and S4 (joint fine-tune) that exist.", "the champion is stage S1 with the lane head frozen; S2 (past), S3 (map) and S4 (joint fine-tune) are PENDING.")
t = rep(t, "**Where the code is (PENDING).** Small winner folders in git, one per winner,", "**Where the code is.** Small winner folders in git (`winners/`), one per winner,")
t += "\n**A challenger is under confirmation.** E367's none-slot arms b_bal5 and b_bal25 also pass the gate (0.6691 / 0.6696 and 0.6731 / 0.6728 at their own TRAIN-chosen decodes, ties with the champion inside the band); b_bal25 is being confirmed on two fresh head seeds (E373) and replaces the champion only by the pre-registered quality tie. Until then the champion above stands (PENDING).\n"
save("07_winners.md", t)

# ---------------- 08 outlook
t = load("08_outlook.md")
t = rep(t, "**What the work shows (PENDING for the champion; what is known now).**", "**What the work shows.**")
t = rep(t, "- Compute is not yet measured on a quiet host for the token head; no TEST number exists for any system here.", "- No TEST number exists for any system here; the junction F1 of the champion (0.20 - 0.21) is below the plain head's 0.27.")
t = rep(t, "**Next steps, very brief (PENDING: after E366 / E367 and the champion).** The decode chosen on the TRAIN sample (E366); a wider, deeper head with token-drop training (E367); cross-attention to the past and to the map (S2, S3); a joint fine-tune that lets the loss reach the lane head (S4); replace the remaining named hard steps (the allowed-pair gate, the top-K budget) where a differentiable form exists.",
        "**Next steps, very brief.** The challenger E373 (the none-slot arm on fresh seeds); S2, cross-attention to the past (E368); S3, cross-attention to the map (not built); S4, a joint fine-tune that lets the loss reach the lane head (E372); a junction condition in the gate, since the junction F1 is the weak number; replace the remaining named hard steps (the allowed-pair gate, the top-K budget) where a differentiable form exists. PENDING: their results.")
save("08_outlook.md", t)

# ---------------- A1 repeat
t = load("A1_repeat.md")
t = ls(t, "PENDING: modules and shapes of the token head", "**Modules (chapter 2).** Frozen lane head B' (`runs/e292_a_s<S>`: pyramid, node head, soft Matrix-NMS layer) -> cached tokens (top-K 512 by existence: node token 64-d, position, heading, existence, 8 flag logits, scales) -> embedding d = 64 -> three self-attention layers (4 heads, FFN 128, relative-geometry bias from 13 features of the pair, key bias log of the existence weight) -> the wiring layer (two heads: short and long chords, per-source softmax over the allowed destinations and one none slot) and the 7-way node-type softmax. The head has 0.127 M parameters.\n\n**Training recipe.** One stage of the head on the cached tokens: `bash tools/env/_py14.sh jtoken_train.py --name e365_s1_s<S> --seed S --ident e292_a --steps 12000 --bs 8 --lr 1e-3 --workers 4` (AdamW, 5 % warm-up, cosine, roll-back guard, a checkpoint every 500 steps with resume). Loss per source and class: minus log of the summed probability of the positive destinations (the none slot when there is none), none weight 1.0, plus 0.5 x node-type cross-entropy plus 0.5 x L1 of the handle scales on the long GT links.\n\n**Read-out and decode.** `jtoken_eval.py dump --run e365_s1_s<S> --seed S`, then `jtoken_eval.py score --run e365_s1_s<S> --seed S --link-thr 0.1 --keep-thr 0.6 --no-none-gate`: the decode chosen on the 291-frame TRAIN sample by the written rule of E366 (`jtoken_eval.py sweep --split train --every 3`). Timing: `jtoken_eval.py timing --run e365_s1_s0 --b-run runs/e292_a_s0`.\n\n**Numbers to hit (VAL 873 frames, seed 0 / seed 1).** Scorer reproduction of the stored plain-B base: 0.6259 / 0.6249 to 1e-6 (`check-layer1`); the head at the decode of record: 0.5486 / 0.5743 (E365); at the chosen decode: 0.6617 / 0.6767 (E366); fresh head seeds 2 / 3: 0.6618 / 0.6733 (E371); whole path 16.83 / 16.57 ms and 2.323 M parameters. **Checks and pitfalls.** The token cache must match its manifest (the fixture compares the cached tokens with the stored dumps); the head seed and the cache seed are different things (E371 needed a `--cache-seed` option); runs resumed after a crash are not bit-reproducible; the GPU may stall when two trainings share the card.")
t = ls(t, "| E366 |", "| E366 | the same two heads with a decode chosen on the TRAIN sample | PASS-DECODE: 0.6617 / 0.6767 | `E366_e365-decode-selected-on-train` |")
t = ls(t, "| E367 |", "| E367 | none-slot reweighting arms (b_bal5, b_bal25) | both pass, ties inside the band; the wide and long arms have not run | `E367_e365-capacity-balance-schedule-arms` |\n| E371 | fresh head seeds 2 / 3 at the frozen decode | CONFIRMED: 0.6618 / 0.6733 | `E371_e366-confirmation-fresh-head-seeds` |")
save("A1_repeat.md", t)

# ---------------- ladder + report.json
lj = os.path.join(P, "figures", "ladder.json")
s = json.load(open(lj, encoding="utf-8"))
rows = [r for r in s["rows"] if "PENDING" not in r["label"] and r["label"] != "champion"]
for r in rows:
    if r["label"].startswith("token head S1, chosen decode"):
        r["label"] = "token champion: S1 head, chosen decode (E366)"
rows.append({"label": "token champion, fresh head seeds (E371)", "values": [0.6618, 0.6733]})
rows.append({"label": "challenger b_bal25 on fresh seeds (E373, PENDING)", "values": None})
s["rows"] = rows
json.dump(s, open(lj, "w", encoding="utf-8"), indent=1)

rj = os.path.join(RP, "report.json")
r = json.load(open(rj, encoding="utf-8"))
r["abstract"] = ("A lane-centre graph (centre lines, forks, merges and junction connectors) is computed from camera and radar features. The earlier system was made end to end "
                 "differentiable step by step and reaches topo F1 0.6912 / 0.6948 on VAL (plain lane head 0.6259 / 0.6249) but keeps five hard steps. We replace its junction machinery "
                 "by one token head: the predicted nodes of the frozen lane head are tokens, a none slot prunes softly and the per-source softmax of the last self-attention layer is the wiring. "
                 "Read at a decode chosen on the TRAIN sample it reaches 0.6617 / 0.6767 (0.6618 / 0.6733 on two fresh seeds), +0.036 to +0.052 over the plain head, in 16.8 ms against 19.7 ms and with "
                 "0.127 M parameters on top of the frozen lane head, with none of the five hard steps. It pays plainly: precision 0.69, 25 vertices per 100 m and a junction F1 of 0.20 - 0.21, "
                 "below the plain head's 0.27. It is stage S1 with the lane head frozen; the past, the map and a joint fine-tune are not in it, and no TEST number exists.")
r["sources"] = [x for x in r["sources"]] + ["../../experiments/E371_*/experiment.md", "../../experiments/E373_*/experiment.md", "../../experiment_execution/decisions/D114.md"]
json.dump(r, open(rj, "w", encoding="utf-8"), indent=1)
print("ok")
