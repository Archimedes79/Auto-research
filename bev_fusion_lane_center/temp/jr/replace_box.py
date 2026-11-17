"""Scratch (run once): replace the hard-step table of the E2E box of 00a_start.md by the compact list of the gap table (D109)."""
import os
P = os.path.join(os.environ["BFLC_ROOT"], "results", "Ergebnis_Spurmitte_2026-11-15", "parts", "00a_start.md")
L = open(P, encoding="utf-8").read().split("\n")
a = next(i for i, l in enumerate(L) if l.startswith("**The named hard steps on the output"))
b = next(i for i, l in enumerate(L) if l.startswith("| **snap geometry**"))
new = """**The named hard steps on the output, what replaces each, where that stands** (labels R1 - R8, T1 - T3 as in the gap table `experiments/E2E_gap_table.md` and card E344; all in the champion):

| step | what it does | replacement and where that stands |
|---|---|---|
| **R1 snap wiring** | joins each connector end to the one nearest delivered piece end within 4 m / 45 degrees (E311) | E325 learned wiring head (545 parameters): KEEP on the E312 recipe (+0.0023 / +0.0032 topo F1 over the rule); its choice is still an argmax; the transfer to E324 s, E350, built as phase A only, nothing trained. Next card: E350 v3, trained and read out on the ATTACHED ends (control PRIMARY 0.6912 / 0.6948) |
| **R2 close-node step** | drops a node within 1.5 m of a node with higher existence (E275) | E277 soft step: CUT by its guards; E279 (1.25 m, pyramid unfrozen): KEEP on B alone, not transferred; E292 Matrix-NMS layer: TIE, kept and refined, not usable (-0.0265 / -0.0231 topo F1 against its control); none in the champion |
| **R3 B's candidate selection** | top-K 512 nodes, +-4-cell window, top-64 x top-64 long links, 8 - 100 m, 75 degree cone, random cap 80 000 pairs | none tried; what the limits cost is not measured (a CPU ceiling check is proposed) |
| **R4 section argmax, one link per node pair, long-link cover drop** | hard choice per section and per pair; a long link of B is dropped when a connector's ends lie within 4 m of its ends | none tried; price not measured |
| **R5 keypoint peaks and top-K 32** | 3 x 3 max-pool with a peak minimum selects the ends, at most 32 per kind (K binds in 2 % of the VAL frames) | E347 (ends on B's 2 m grid): both arms failed, closed; a threshold alone without the peak step is 0.127 / 0.125 worse in keypoint F1; needs a 1 m recipe (Q-Lane, E344) |
| **R6 pair candidate mask** | a pair of ends may connect only if turn angle, minimum chord and forward chord hold | none tried; share of GT pairs it excludes not measured |
| **R7 attach gate** | the attachment sees only the nearest 8 nodes within 4 m / 45 degrees with existence above 0.05 | the attachment itself is learned (E320 / E324 s); the gate's ceiling is not measured |
| **R8 group / rank bookkeeping** | the order rule that gives the pair head its group and rank features | none wanted: by design no gradient (D82), stays named |
| **T1 - T3 thresholds** | node existence >= 0.5, link p >= 0.5, connector p >= 0.1 | none: the consumer reads a probability above a threshold; a threshold-free report is a proposed card, it removes no step |
| **snap geometry** (only in the snapped record read-out) | the connector curve starts at the wired piece end | not part of the end-to-end read-out; the learned attachment recovers 25 % / 34 % of it, the rest +0.0069 / +0.0062 topo F1 is geometry (E337); a sharpening lever is a stub, not scheduled |

**Douglas-Peucker is not a delivered step.** The output stage `simplify_dp` is off by default (PROBLEM.md, E344); 0.3 m is only the metric's own vertex basis (vertices per 100 m).""".split("\n")
L[a:b + 1] = new
open(P, "w", encoding="utf-8").write("\n".join(L))
print("ok", a, b)
