"""one-off patch of build_total_report.py (2026-11-13 09:00): E320 measured (CUT by a guard, tie with the snap), E322 gate G0 (no headroom for a learned selection), E323 (the duplicates are not the curve count)."""
import os

p = os.path.join(os.environ["BFLC_ROOT"], "temp", "jr", "build_total_report.py")
s = open(p, encoding="utf-8").read()


def rep(old, new):
    global s
    assert old in s, old[:70]
    s = s.replace(old, new, 1)


rep('"<b>E320 soft attachment</b>: a softmax over the delivered piece ends near each connector end (with a dustbin), trained by the geometry loss E317 on the final curve; the snap goes when the soft version is within 0.01 topo F1 of it",',
    '"<b>E320 measured (VAL, 2026-11-13): the learned attachment replaces the geometry of the snap</b> (topo F1 of the attached read-out with the snap\'s one-piece wiring: a - c = -0.0021 / +0.0025, a tie with the snap; a - u\' = +0.0097 / +0.0130) but the card is CUT by its duplicates guard (dup_frac +0.0184 on seed 0). E323: the excess is not the curve count. E322 gate G0: a learned selection of the wired piece has no headroom (the one-piece rule is a count effect and a named selection on the output, allowed by D55)",')
rep('["E320", "soft attachment of the connector ends to the delivered pieces: replaces the snap", "yes", "small", "designed; code and card still to be written; the highest-value step"],',
    '["E320", "learned attachment of the connector ends to B\'s nodes + the E317 geometry loss: replaces the geometry of the snap", "yes (the one-piece wiring on the output stays)", "+9.6 k parameters, +3.9 MMACs", "MEASURED: CUT by the duplicates guard (dup_frac +0.0184 on seed 0, +0.0093 on seed 1; topo F1 ties the snap); E323: not the curve count"],\n            ["E322", "soft wiring: a learned choice of the delivered piece a connector end is wired to", "yes", "small, not built", "gated card: G0 (oracle selection) shows no headroom; closed at the gate; the head stays designed (equal-accuracy bar), one question for Mirko: E2E replacement of the output selection against simplicity"],\n            ["E323", "operating point of the read-out: is the duplicate excess of E320 the curve count?", "n/a (read-out only, CPU)", "0", "answered: no, E320 CUT stands"],\n            ["E331 - E334", "simplification arms (D84 point 4): evidence off, no paint branch, slim maps head, one joint stage", "removals", "lower cost", "E331 running since 2026-11-13 08:30 (rule: loss <= 0.01 snapped topo F1 on both seeds adopts the removal), E332 - E334 queued after it"],')
rep("the snap, a rule on the output, carries most of the topology gain: E320 replaces it", "the snap, a rule on the output, carries most of the topology gain; E320 replaced its geometry (CUT by a guard)")
rep("E317 / E318 / E319 / E320 are the queued experiments.</li>", "E320 (learned attachment) is measured (CUT by a guard, a tie with the snap), E322 (soft wiring) closed at its gate G0, E317 / E318 / E319 are held (D85).</li>")
rep("<b>Not done:</b> the experiments of the night (E315 - E320) are queued or running and are not in the numbers; no later design has been run on TEST.</li>", "<b>Not done:</b> no later design (E320 and after) has been run on TEST; E315 - E323 are VAL results (E315, E316 TIE, E320 CUT by a guard, E322 / E323 read-outs); the simplification arms E331 - E334 are running.</li>")
rep("The next gain is not another head but a differentiable replacement of the snap (E320): the unsnapped row is what a fully differentiable read-out gives today.", "The snap's geometry has a differentiable replacement (E320: a tie with the snap on topo F1, CUT by a duplicates guard); the one-piece wiring stays a named selection on the output (E322 gate G0: a learned choice has no headroom). The next gains come from simplification (E331 - E334), the free 6000-step pair training (E316, +0.007) and the geometry loss (the E320 recipe's own snapped read-out is +0.005 / +0.006 over E312).")
rep("So the attachment of E320 (geometry) can recover at most about a third of the gap, and the wiring rule (argmin over pieces) is the step to make soft next (E322: attach to delivered piece ends with a dustbin; E320 pre-registers a tight-radius wiring read-out as the first test).</div>",
    "So the attachment of E320 (geometry) could recover at most about a third of the gap, and it did (table below). The wiring is not a which-piece effect: E322 gate G0 (oracle selection at the matched GT end) gains {G0TXT}; the one-piece rule is a COUNT effect (one piece instead of every piece in reach), a named selection on the output that D55 allows; a learned head stays designed with an equal-accuracy bar and is the one question left for Mirko (end-to-end replacement of the output selection against a simple system).</div>\n<h3>E320: the learned attachment, read-outs on VAL</h3>\n{e320_table()}\n<p class=\"cap\">Rule of the card (written before the run): KEEP when the primary read-out reaches u' + 0.005 on both seeds with the guards held; the duplicates guard breaks on seed 0 (dup_frac +0.0184 against the limit +0.01), so the verdict is CUT by the written rule. Bootstraps (22 scenes): a - c -0.0021 [-0.0079, +0.0041] / +0.0025 [-0.0078, +0.0106], a tie with the snap; junction F1 -0.0256 / -0.0137, significantly lower but inside the guard. The tight 1.5 m / 1.0 m wiring fails on both seeds: a contact threshold does not replace the one-piece rule. E323 reads E312 and E320 at equal curve counts: the duplicate excess is not the curve count.</p>")
rep("until E320 and a soft close-node step are in, the delivered graph is end to end only up to these named rules.", "the delivered graph is end to end only up to these named rules (the snap's geometry has a differentiable replacement, E320, which a duplicates guard rejected; the one-piece wiring and the close-node step stay named selections on the output until a soft close-node step, E292, is measured).")

# the function e320_table() and the G0 text, before the best-approaches block
rep("# ------------------------------------------------------------------------------------------------------------------ best approaches", '''def e320_table():
    rows = []
    spec = (("E312 snapped (c, the card of record)", "e312_p_s{s}", "ends_card_snap_thr0.1.json"), ("E312 snap-nogeom (u')", "e312_p_s{s}", "ends_card_snap_nogeom_thr0.1.json"),
            ("<b>E320 primary: attached ends + one-piece wiring</b>", "e320_a_s{s}", "ends_card_snap_nogeom_attach_thr0.1.json"), ("E320 own keypoints snapped", "e320_a_s{s}", "ends_card_snap_thr0.1.json"),
            ("E320 attached, proximity wiring (secondary)", "e320_a_s{s}", "ends_card_attach_thr0.1.json"), ("E320 attached, tight wiring 1.5 m", "e320_a_s{s}", "ends_card_attach_r1.5_thr0.1.json"),
            ("E320 attached, tight wiring 1.0 m", "e320_a_s{s}", "ends_card_attach_r1.0_thr0.1.json"))
    for s in (0, 1):
        for lab, run, f in spec:
            c = J(f"runs/{run.format(s=s)}/{f}")
            if not c:
                rows.append([str(s), lab] + ["pending"] * 6)
                continue
            h = c["numbers"]["head"]
            rows.append([str(s), lab, f3(h["topo_F1"]), f3(h["topo_P"]), f3(h["junction_F1"]), f3(h["lanes_found"]), f3(h["dup_frac"]), f3(h["F1@1.0"])])
    return tbl(["seed", "read-out (p >= 0.1)", "topo F1", "topo precision", "junction F1", "lanes found", "dup_frac", "F1@1.0"], rows, num=(0, 2, 3, 4, 5, 6, 7))


G0TXT = "-0.0007 / +0.0013 topo F1 over the nearest-piece rule (E312 s0 / s1, card E322); the same piece is chosen in 84 % of the ends"


# ------------------------------------------------------------------------------------------------------------------ best approaches''')
open(p, "w", encoding="utf-8").write(s)
print("patched")
