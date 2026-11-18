p = "experiments/E371_e366-confirmation-fresh-head-seeds/experiment.md"
s = open(p, encoding="utf-8").read()


def rep(a, b):
    global s
    assert s.count(a) == 1, a[:70]
    s = s.replace(a, b)


rep("Card of 2026-11-17 (version 1, 03:15 system clock, before any run;",
    "Card of 2026-11-17 (version 2, 03:40 system clock, after the first critic pass (fresh ar-critic, cards: REVISE, 7 findings, taken in: the option must be tested before the launch; the consequence of NOT CONFIRMED; the champion definition against D113; the B' caches 0 / 1 said; what is reported; the prediction per cache; JSON and cost); before any run;")
rep("E367's champion rule (written before) asks for a confirmation of a passing recipe on two fresh head seeds before \"a champion exists\" is reported.",
    "E367's champion rule (written before) asks for a confirmation of a passing recipe on two fresh head seeds before \"a champion exists\" is reported (E366 / E367 call the option `--cache-seed`; it is realised as `--head-seed`: `--seed` stays the B' cache seed).")
rep("**The question:** does the FROZEN recipe", "**Statement:** the E366 result is a property of the recipe, not of the two heads it was selected on, on the B' caches 0 / 1 (a new B' seed would need two pyramid trainings and new plain-B floors: another claim). **The question:** does the FROZEN recipe")
rep('"ref": {"recipe": "E366 (head seeds 0 / 1 on the caches 0 / 1): VAL topo F1 0.6617 / 0.6767; the verdict is by hand (decide.py not used)"}',
    '"ref": {"recipe": "E366 (runs E366.val_s0 / val_s1: head seeds 0 / 1 on the caches 0 / 1): VAL topo F1 0.6617 / 0.6767; the verdict is by hand (decide.py not used)"}')
rep('"run_ids": ["E371.cf_s0", "E371.cf_s1"]', '"run_ids": ["E371.cf_s0", "E371.cf_s1", "E371.read_s0", "E371.read_s1"]')
rep('"budget_gpu_h": 1.0,', '"budget_gpu_h": 1.5,')
rep("`--timeout` 130 minutes (12000 x 0.5 s x 1.3)\",",
    "`--timeout` 130 minutes (12000 x 0.5 s x 1.3); a resumed run is marked not bit-reproducible (as E367); a crashed read-out is re-run unchanged, never with another cell\",")
rep("NOT CONFIRMED = G-topo or G-found fails on either fresh head (a stricter reading than a MISS: a confirmation must hold on both). **\"A champion exists\" (D113) = E366 PASS-DECODE (quality) + E371 CONFIRMED + G-cost confirmed on a quiet host.**",
    "NOT CONFIRMED = G-topo or G-found fails on either fresh head (a stricter reading than a MISS: a confirmation must hold on both). A fresh head below a floor by less than the band 0.01 is NOT CONFIRMED, labelled 'at the floor'. No third head is added after the read-out; the rule is not swapped afterwards for a pooled or four-head mean. NOT CONFIRMED is reported as 'the gate holds on k of 4 heads'; E366 then stays the best measured S1 read-out, not a champion, and the champion claim waits for a recipe that passes its own confirmation (E367's arms). The JSON rule is NOT the verdict (decide.py is not used). **'A champion exists' = D113 point 4 (quality, G-cost, G-simple on both seeds) plus E367's confirmation rule (stricter than D113) = E366 PASS-DECODE (quality) + E371 CONFIRMED + G-cost ms on a quiet host + G-simple; reported as 'staged, B' frozen' (E365, until S4) and as the S1 stage of D113 point 1's design (no past or map attention yet). If an E367 arm passes its own confirmation with a higher mean, that arm is the champion (E367's rule); E371 decides only whether one exists.**")
rep("Reported: the four heads (seeds 0 - 3) side by side,",
    "Reported: the paired cf - ref interval per cache (the first spread due to the head seed alone); the E366 rule run on each fresh head's own TRAIN sample (CPU sweep), written AFTER its VAL score and never used for the verdict (does the rule pick the frozen cell 0.1 / 0.6 / off for a fresh head? the cell sits next to the duplicate guard: TRAIN dup_frac 0.0435, the neighbour 0.0501; VAL dup_frac of E366 0.055 / 0.054); the note that VAL 873 was read by E365 / E366 / E367 / E369 / E370 (the confirmation removes the selection of the head instance, not the reuse of VAL across cards); the four heads (seeds 0 - 3) side by side,")
rep("**Prediction (guesses, with windows):** VAL topo F1 0.665 (0.635 - 0.69) per fresh head; CONFIRMED about 75 % (the margin of E366 over the floors is +0.026 / +0.042; the seed spread of the S1 heads was 0.026 at the record decode).",
    "**Prediction (guesses, with windows):** VAL topo F1 on cache 0 about 0.655 (0.63 - 0.68), on cache 1 about 0.670 (0.645 - 0.695); |cf - ref| per cache about 0.01 (0 - 0.03); the tightest condition is F1@1.0 on cache 0 (E366: 0.8359 against the floor 0.8182: +0.018), then topo F1 on cache 0 (+0.026); the spread between the two E366 heads at this decode was 0.015 (cache and head seed mixed); CONFIRMED about 75 %.")
rep("- **Statement:** the E366 result is a property of the recipe, not of the two heads it was selected on. **Not claimed:** that the decode is optimal for other heads; anything about S2 / S3 / S4 / TEST.",
    "- **Not claimed:** that the decode is optimal for other heads; robustness to a new B' seed (a new gate, another claim); anything about S2 / S3 / S4 / TEST.")
rep("| option | `--head-seed` (coder; default bit for bit unchanged) | coder | test green, committed |",
    "| option | `--head-seed` wired into `T.seed_all`, `T.fit(seed=)` and `make_loss` (the loader keeps `--seed`) | coder | tests green: (a) `--seed 0 --head-seed 2` gives other step-0 weights and another first-batch frame order than `--seed 0`; (b) the cache dir is still seed 0's; (c) without `--head-seed` the head is bit for bit the E365 S1 head (the equivalence test extended to the new prior / teacher / init / snapshots / past_block defaults); (d) the log prints both seeds and 127 131 trainable parameters; committed; the launch waits for this row |")
rep("| read-out | dump (TRAIN every 3, VAL), score at the frozen decode (VAL; TRAIN sample as a diagnosis) | `run.py --id E371.read_s<S>` | cards |",
    "| read-out | dump (TRAIN every 3, VAL), score at the frozen decode (VAL; the TRAIN sample as a diagnosis), then the TRAIN sweep of each fresh head (diagnosis, after the VAL score) | `run.py --id E371.read_s<S>` | cards |")
open(p, "w", encoding="utf-8").write(s)
print("ok")
