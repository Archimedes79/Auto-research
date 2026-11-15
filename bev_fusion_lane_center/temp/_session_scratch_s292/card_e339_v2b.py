def rd(p):
    return open(p, encoding="utf-8").read()


def sub(s, a, b):
    assert s.count(a) == 1, a[:80]
    return s.replace(a, b)


p = "experiments/E339_narrow-head-with-teacher/experiment.md"
s = rd(p)
s = sub(s, "summed over the cells and divided by the teacher's probability mass (the number of positives, as CenterNet / FCOS normalise; a mean over the 10 000 empty cells would dilute the gradient of the ends to nothing),",
        "summed over the cells and divided by the number of peak cells of the teacher (heat probability above 0.3, the same mask as the heading term: the number of positives, as CenterNet / FCOS normalise; a mean over the 10 000 empty cells would dilute the gradient of the ends to nothing, and the probability mass, the first draft, makes the scale depend on the teacher's background calibration: about 146 at a background logit of -7, about 1000 at -5, against 300 - 480 of peak mass per batch: critic),")
s = sub(s, "the ratio of the distillation term to the ground-truth heat loss is in the log every step (`k_bce`, `j_heat`).",
        "both terms are in the log every step (`k_bce`, `j_heat`). **The weight 1.0 is not an equal pull:** per peak cell the distillation gradient is about (student - teacher) / (number of peak cells), the ground-truth heat BCE is a mean over about 160 000 cells with the positive weight 20, so the soft term probably dominates the heat gradient (estimate, not measured: the logged values have different normalisations and do not show the gradient ratio); the arm is then mostly imitation of the teacher plus the ground-truth dice, headings and zone / core / paint terms. One value, written before any number.")
s = sub(s, "The matched control is E333's plain arm, which exists because this card runs only after it.",
        "The matched control is E333's plain arm. It does not exist when E333 is INCOMPLETE (its stage-1 kill) or its card is missing, which day17 also sends here: then the \"+0.010 over plain\" claim is impossible and only the adoption against E312 is judged. A NOT adopted of E333 that comes from a guard (dup_frac, junction F1) with topo F1 inside the band is not \"a loss the teacher can win back\": it is reported next to the result.")
s = sub(s, "\"E339 `--teacher`", "\"E339 `--teacher`") if False else s
s = sub(s, "four mutants (no detach, soft heading weight, pos-weighted BCE, reversed heading sign) are all killed (scratch run, `temp/_session_scratch_s292/mutants_distill.py`).",
        "eleven mutants (no detach, soft heading weight, pos-weighted BCE, reversed heading sign, exit heat dropped, exit heading read from the entry channels, mask threshold 0.05 and 0.9, mean instead of sum / peaks, mass instead of peaks, peaks of the entry channel only) are all killed by a fixed one-frame pair with both kinds and a hand-computed loss value (scratch run, `temp/_session_scratch_s292/mutants_distill.py`).")
s = sub(s, "A second look of the critic on this version is asked for together with the E340 card.",
        "Second pass of the same critic: card REVISE (small: the normaliser was the probability mass, the weight is not an equal pull, the case without a plain control), E340 REVISE (two `--pred` strings copied from E333), code ACCEPT with a test gap (six mutants survived: the exit side was never driven, the 0.3 boundary and the normaliser were not pinned), scheduler ACCEPT. Answered above and in the code (normaliser = peak cells, the fixed-pair test, both prediction strings set to the card's numbers).")
open(p, "w", encoding="utf-8", newline="\n").write(s)
print("ok")
