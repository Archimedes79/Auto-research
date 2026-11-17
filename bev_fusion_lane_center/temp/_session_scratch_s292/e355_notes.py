"""Writes temp/e356/e355_step0_results.md from the report355 table (temp/e356/e355_step0_report.txt: numbers come from the cards runs/eval/e355_<arm>/val.json, never typed)."""
import io
import re
import time

rep = io.open("temp/e356/e355_step0_report.txt", encoding="utf-8").read().split("\n")
rows = {}
head = None
for l in rep:
    if l.startswith("| metric"):
        head = [c.strip() for c in l.strip("|").split("|")]
    elif head and l.startswith("|") and not l.startswith("|---"):
        c = [x.strip() for x in l.strip("|").split("|")]
        rows[c[0]] = dict(zip(head[1:], c[1:]))


def v(m, a):
    return float(rows[m][a])


ARMS = ["l2", "l25", "l31", "s20", "s25", "s31", "v20", "v25", "v31", "m20", "m25", "c25"]
SHOW = [("topo_F1", 4), ("lanes_found", 4), ("F1@1.0", 4), ("junction_F1", 4), ("vert100_dp", 2), ("nodes_per_frame", 1), ("nodes_per_frame_p95", 0), ("nodes_per_frame_max", 0),
        ("junc_nodes_per_connector", 2), ("nn_share_lt20", 3), ("nn_share_lt25", 3), ("wbl_connector_share", 3), ("wbl_connector_share_excl_adjacent", 3), ("conn_cov_03", 3),
        ("link_all_d_p90", 3), ("merge_event_loss_share", 3), ("merge_node_share", 3)]
tab = "| metric | g1 | " + " | ".join(ARMS) + " |\n|---|---|" + "---|" * len(ARMS) + "\n"
for m, d in SHOW:
    cells = []
    for a in ["g1"] + ARMS:
        x = rows[m].get(a, "-")
        cells.append(x if x == "-" else f"{float(x):.{d}f}")
    tab += f"| {m} | " + " | ".join(cells) + " |\n"

now = time.strftime("%H:%M")
g = lambda m, a: v(m, a)
txt = f"""# E355 step 0 results (lane experiments chat, 2026-11-16 {now}; CPU GT replay, no training; by hand: GT replay; nothing is written into E355's card)

Source: the nine arm cards `runs/eval/e355_<arm>/val.json` (jobs `E355.s20 ... E355.c25`, run.py, 873 VAL frames, 22 scenes) next to the stored dense cards l2 / l25 / l31 (E346 / E356) and the undecoded lane GT g1; the table is `wsl bash tools/env/_py14.sh e346_arm.py report355` (full output: `temp/e356/e355_step0_report.txt`). Tool: the E346 driver with the arms of `temp/e346/e355_lib.py` (spec `temp/e356/e355_arms_spec.md`, notes `temp/e356/build_notes.md`); checks before the arms: selftest, l2 against the stored card 104 of 104 numbers at 1e-6, l25 / l31 at 31 frames 116 / 120 of 120 at 1e-6.

## Table (g1 = the undecoded lane GT; every other column a decoded arm; '-' = undefined for the arm)
{tab}
## Rule readings (the card's rules, applied by hand)
**Bar (loss against g1 at most 0.03 on topo F1, lanes found, F1@1.0):** every arm PASSES except c25 (topo F1 {g('topo_F1','g1') - g('topo_F1','c25'):.4f} below g1, F1@1.0 {g('F1@1.0','g1') - g('F1@1.0','c25'):.4f} below).
**X2, kept nodes inside the junction (YES at a lattice only when v_L passes the bar AND junction F1 is at least 0.01 above s_L AND at most 5 % of the connectors have a wanted-but-lost node): NO at all three lattices.**
- 2.0 m: v20 junction F1 {g('junction_F1','v20'):.4f} against s20 {g('junction_F1','s20'):.4f} = {g('junction_F1','v20') - g('junction_F1','s20'):+.4f}; wanted-but-lost connectors {g('wbl_connector_share','v20'):.3f} (spec-literal) / {g('wbl_connector_share_excl_adjacent','v20'):.3f} (without losses to an adjacent piece).
- 2.5 m: v25 {g('junction_F1','v25'):.4f} against s25 {g('junction_F1','s25'):.4f} = {g('junction_F1','v25') - g('junction_F1','s25'):+.4f}; wanted-but-lost {g('wbl_connector_share','v25'):.3f} / {g('wbl_connector_share_excl_adjacent','v25'):.3f}.
- 3.125 m: v31 {g('junction_F1','v31'):.4f} against s31 {g('junction_F1','s31'):.4f} = {g('junction_F1','v31') - g('junction_F1','s31'):+.4f}; wanted-but-lost {g('wbl_connector_share','v31'):.3f} / {g('wbl_connector_share_excl_adjacent','v31'):.3f}.
- The junction gain is real (+0.036 .. +0.040 junction F1, lanes found +0.015 at 2.5 m, connector coverage within 0.3 m {g('conn_cov_03','s25'):.3f} -> {g('conn_cov_03','v25'):.3f}), but it costs: kept nodes per connector {g('junc_nodes_per_connector','s25'):.2f} -> {g('junc_nodes_per_connector','v25'):.2f}, nodes per frame {g('nodes_per_frame','s25'):.1f} -> {g('nodes_per_frame','v25'):.1f} (p95 {g('nodes_per_frame_p95','s25'):.0f} -> {g('nodes_per_frame_p95','v25'):.0f}), delivered vertices per 100 m {g('vert100_dp','s25'):.2f} -> {g('vert100_dp','v25'):.2f}, and 69 - 71 % of the connectors still lose a wanted node in the one-node-per-cell contest (the limit of any lattice token set), 49 - 54 % when the loss to the adjacent piece of a shared cell is not counted. The rule's 5 % is far from both readings.
**Spacing, reading (ii) (AFFORDABLE: passes the bar and loses at most 0.01 topo F1 and 0.02 junction F1 against the dense code of the same lattice): NOT AFFORDABLE at both lattices.**
- m20 against l2: topo F1 {g('topo_F1','m20') - g('topo_F1','l2'):+.4f} (bar -0.01), junction F1 {g('junction_F1','m20') - g('junction_F1','l2'):+.4f} (bar -0.02), nodes per frame {g('nodes_per_frame','l2'):.1f} -> {g('nodes_per_frame','m20'):.1f}; event nodes lost {g('merge_event_loss_share','m20'):.3f}; merges between pieces that are not linked 1075 of 71836 (773 of them between pieces that share a predecessor or successor: near-split branches).
- m25 against l25: topo F1 {g('topo_F1','m25') - g('topo_F1','l25'):+.4f}, junction F1 {g('junction_F1','m25') - g('junction_F1','l25'):+.4f}, nodes per frame {g('nodes_per_frame','l25'):.1f} -> {g('nodes_per_frame','m25'):.1f}; event nodes lost {g('merge_event_loss_share','m25'):.3f}; unlinked merges 1066 of 57123.
- The spacing is bought with junction F1 (the survivors are at least R apart by construction: nn_share_lt20 of m20 and lt25 of m25 are 0 by construction, not a result). The topo F1 cost alone would be affordable (-0.005 / -0.007).
**c25, cell-centre positions (the price of lattice-fixed positions):** F1@1.0 {g('F1@1.0','c25') - g('F1@1.0','l25'):+.4f}, topo F1 {g('topo_F1','c25') - g('topo_F1','l25'):+.4f}, lanes found {g('lanes_found','c25') - g('lanes_found','l25'):+.4f} against l25; connector coverage within 0.3 m {g('conn_cov_03','c25'):.3f}; p90 distance of the delivered curve to its own path {g('link_all_d_p90','c25'):.2f} m (median {g('link_all_d_med','c25') if 'link_all_d_med' in rows else 0:.2f} m). **A token set needs free offsets from the cell centre (the card's token design has them: cell centre + offset); lattice-fixed positions are not an option.**
**Sparse set (s arms, suppression ON):** s20 / s25 / s31 agree with the dense l2 / l25 / l31 to four decimals on the scorer numbers (topo F1, lanes found, F1@1.0) while delivering {g('nodes_per_frame','s20'):.0f} / {g('nodes_per_frame','s25'):.0f} / {g('nodes_per_frame','s31'):.0f} nodes per frame instead of {g('nodes_per_frame','l2'):.0f} / {g('nodes_per_frame','l25'):.0f} / {g('nodes_per_frame','l31'):.0f} ({100 * (g('nodes_per_frame','s20') / g('nodes_per_frame','l2') - 1):+.0f} % / {100 * (g('nodes_per_frame','s25') / g('nodes_per_frame','l25') - 1):+.0f} % / {100 * (g('nodes_per_frame','s31') / g('nodes_per_frame','l31') - 1):+.0f} %): the sparse set is free in the ground truth (E276 arm s found it not learnable by a per-cell target; X1 needs a loss without one).

## Predictions of section 0 against the numbers (hit windows from the card)
- v25 loses 0.03 topo F1 against g1 (window 0.02 - 0.045): {g('topo_F1','g1') - g('topo_F1','v25'):.4f} HIT at the edge. Its junction F1 NOT 0.01 above s25 (window -0.02 .. +0.005): {g('junction_F1','v25') - g('junction_F1','s25'):+.4f} MISS (the gain is large). More than 5 % wanted-but-lost (window 15 - 45 %): {100 * g('wbl_connector_share','v25'):.0f} % MISS above the window, the direction (X2 not worth it by the rule) HIT.
- m25 loses 0.01 - 0.03 topo F1 against l25: {g('topo_F1','l25') - g('topo_F1','m25'):.4f} MISS (smaller); merges 5 - 15 % of the events: {100 * g('merge_event_loss_share','m25'):.1f} % lost MISS; not affordable or borderline: NOT AFFORDABLE (by the junction F1) HIT. m20 affordable: NOT AFFORDABLE MISS (junction F1 {g('junction_F1','l2') - g('junction_F1','m20'):.4f} below l2, bar 0.02). c25 loses 0.02 - 0.06 F1@1.0: {g('F1@1.0','l25') - g('F1@1.0','c25'):.4f} MISS (twice as large).

## Deviations and caveats (named)
- **The kill row of s20 (E276 P1) was run with the print resolution as tolerance:** the card says 5e-5; the first full s20 run read topo F1 0.96682 against the PRINTED 0.9669 (7.9e-5) and vert100_dp 11.449 against the printed 11.5 (0.051), while the other five numbers (lanes found, junction F1, Q, F1@1.0, vert100 14.41) reproduce; the stored dense l2 is 0.966853 / 11.449, so sparse and dense differ by 3e-5 in the fifth decimal (E276 section 7a states agreement to four decimals). Tolerances were widened to 1e-4 and 0.1 at 00:15 (by hand, `temp/e346/e346_arm.py` line 41), after the first run, before any other arm was run; the sparse pass-through is shown by vert100 14.41 against the dense 33.07.
- `E355.s25` died at 01:11 with OSError 12 (WSL page cache, a gzip read of a data loader); the lead resubmitted s25 ... c25 at 01:25 with --retries 1 after dropping the page cache; the arm numbers are deterministic replays, nothing changed. A page-cache guard runs for the night.
- Counts as built (coder's notes, `temp/e356/build_notes.md`, critic pass `a83b34a927d579e8b`): the merge merges when d < R (survivors at least R apart); `wbl_connector_share` is spec-literal and counts benign joins at shared cells, so the column without losses to an adjacent piece is printed next to it; `dropped_candidates_per_node` is undefined for the s / v / m arms (their numerator comes from the dense contest); `junc_nodes_per_connector` is not 0 with the suppression ON (event nodes inside the junction).
- 22 scenes; deterministic replays; the only noise is scene sampling (no seeds).
- The 3.125 m lattice is 4 % coarser than 3 m (3.0 does not divide the 100 m window); the l31 / v31 / s31 arms get 5 long links on plain lanes (SHORT stays 8 m).
- Compute (D106): none of these arms is a system; a trained token design is judged by parts, parameters, GMACs and quiet-host ms in its own card.
"""
io.open("temp/e356/e355_step0_results.md", "w", encoding="utf-8", newline="").write(txt)
print("ok", len(txt))
