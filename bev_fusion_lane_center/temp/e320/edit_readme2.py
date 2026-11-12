p = "temp/e320/README_eval_head.md"
lines = open(p, encoding="utf-8").read().split("\n")
def append_to(prefix, old_tail, new_tail):
    for i, l in enumerate(lines):
        if l.startswith(prefix):
            assert l.rstrip().endswith(old_tail), (prefix, l[-80:])
            lines[i] = l.rstrip()[: -len(old_tail)] + new_tail
            return
    raise AssertionError(prefix)
append_to("| `qends_eval.py` |", "|", "`make_pairs` (the one place that chooses the keypoint or the attached ends of a dump frame, E320), `add_connectors(wire_pos=)` (the wiring radius, default 4 m), `score` writes `long_links_dropped` and `ends_within_1m_of_a_delivered_piece_end` into the card |")
append_to("| `jends_eval.py` |", "|", "E320 read-outs of an attach run: `score --attach --snap --snap-nogeom --thr 0.1` (the PRIMARY read-out: attached ends, one nearest piece per end, `ends_card_snap_nogeom_attach_thr0.1.json`), `score --attach` (proximity wiring), `score --attach --attach-r 1.5` / `1.0` (tight wiring radius, `ends_card_attach_r<r>_thr<t>.json`) |")
append_to("| `jmaneuver.py` |", "|", "E320 variants: `p0.1a` (attached ends, proximity wiring), `p0.1an` (attached + one nearest piece per end, the primary read-out), `p0.1ar1.5` (attached, wiring radius 1.5 m) |")
append_to("| `jdiag.py` |", "|", "`kperr` also prints the error of the ATTACHED ends (`att_pos`) of an E320 dump next to the keypoint error (E320) |")
open(p, "w", encoding="utf-8").write("\n".join(lines))
print("readme blob ok")
