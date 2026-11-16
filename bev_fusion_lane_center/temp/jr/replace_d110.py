"""Scratch (run once): replace the D110 paragraphs (decision wording) by the open-question wording."""
import os
P = os.path.join(os.environ["BFLC_ROOT"], "results", "Ergebnis_Spurmitte_2026-11-15", "parts")

START = ("**Mirko's criticism and the open question (D110).** His sentence of 2026-11-16, translated: \"we work the whole time in a solution space that does not exist, "
         "and discard everything in the prescribed one\" (original: \"Verstehe ich, aber wir arbeiten die ganze Zeit in einem nicht existierenden Loesungsraum und verwerfen alles in dem vorgegebenen Loesungsraum\"). "
         "The plain fact: for eleven days the headline, the champion and the reference of most cards was a system with rule steps (snapped read-out, nearest-piece wiring, hard close-node step), "
         "and the end-to-end candidates were judged against it; most were cut or not adopted (E274, E277, E318, E319, E347, E320 at first, E292 and E354 as ties, E357). "
         "**Proposal, his word needed (nothing of it is in force):** the end-to-end system (0.6912 / 0.6948; strict line attach-only 0.6616 / 0.6646) is the reference and control of every new card, the snapped record only an upper reference line; "
         "an end-to-end candidate is not cut for scoring below a rule-step system but judged by whether it learns, by its result against the end-to-end control and by its price; "
         "a learned or soft replacement of a named hard step that ties or beats the end-to-end control is adopted. The old verdicts stay as written. "
         "Until he answers, the lane chat prepares a re-judgement table of the cut and not-adopted end-to-end candidates against the end-to-end control as an analysis of stored cards (`experiments/E2E_gap_table.md`; in progress, no numbers yet) "
         "and the end-to-end stack as candidate cards (the learned wiring head on the attached ends, E350 v3; Matrix-NMS instead of the close-node step, E292), each with a critic pass first; verdict rules of new cards follow D87 (an end-to-end replacement without an alternative is kept and refined, not cut); adopting a tie is not applied.")

OPEN = ("**0. D110, an open question (Mirko's criticism of 2026-11-16 about 09:50, after the answer to D109; `decisions/D110.md`; his answer is awaited).** His sentence, translated: \"we work the whole time in a solution space that does not exist, and discard everything in the prescribed one\". "
        "The plain fact: for eleven days the headline and the reference of most cards was a system with rule steps, and the end-to-end candidates were judged against it; most were cut or not adopted (E274, E277, E318, E319, E347, E320 at first, E292 and E354 as ties, E357). "
        "**Proposal, his word needed:** (1) the end-to-end system (0.6912 / 0.6948; strict line attach-only 0.6616 / 0.6646) is the reference and control of every new card, the snapped record an upper reference line only; "
        "(2) an end-to-end candidate is not cut for scoring below a rule-step system, it is judged by whether it learns, by its result against the end-to-end control and by its price; "
        "(3) a learned or soft replacement of a named hard step that ties or beats the end-to-end control is adopted; (4) finished tokens are never rewritten, a labelled re-judgement table is an analysis of stored cards. "
        "His question in one line (German): \"Soll der Massstab so geaendert werden (E2E-System als Referenz, E2E-Kandidaten nicht verwerfen weil sie unter einem Regelschritt-System liegen, Gleichstand beim Ersatz eines harten Schritts uebernehmen)? ja / nein / anders\". "
        "Meanwhile (inside D88 / D109, nothing new decided): the lane chat prepares the re-judgement table as an analysis (in progress, no numbers yet) and the end-to-end stack as candidate cards (E350 v3, E292 Matrix-NMS), each with a critic pass first; verdict rules of new cards follow D87; point 3 is not applied.")

for fn, head, new in (("00a_start.md", "**What changes (D110", START), ("08_open.md", "**0. D110", OPEN)):
    f = os.path.join(P, fn)
    L = open(f, encoding="utf-8").read().split("\n")
    i = next(k for k, l in enumerate(L) if l.startswith(head))
    L[i] = new
    open(f, "w", encoding="utf-8").write("\n".join(L))
print("ok")
