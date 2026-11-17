"""one-off patch of build_total_report.py (2026-11-14 02:00): E324 arm s (the E320 recipe at 6000 pair steps): the best read-outs so far, a system-of-record candidate."""
import os

p = os.path.join(os.environ["BFLC_ROOT"], "temp", "jr", "build_total_report.py")
s = open(p, encoding="utf-8").read()


def rep(old, new):
    global s
    assert old in s, old[:70]
    s = s.replace(old, new, 1)


rep("def e320_table():", '''def e324_table():
    spec = (("E312 snapped (c, the system of record)", "e312_p_s{s}", "ends_card_snap_thr0.1.json"), ("E316 snapped (E312 recipe, 6000 steps)", "e316_p_s{s}", "ends_card_snap_thr0.1.json"),
            ("E320 primary (attached ends, one-piece wiring)", "e320_a_s{s}", "ends_card_snap_nogeom_attach_thr0.1.json"),
            ("<b>E324 s primary (attached ends, one-piece wiring)</b>", "e324_s_s{s}", "ends_card_snap_nogeom_attach_thr0.1.json"),
            ("<b>E324 s own keypoints, snapped (the E312 read-out type)</b>", "e324_s_s{s}", "ends_card_snap_thr0.1.json"),
            ("E324 s own keypoints, one-piece wiring, no snap geometry", "e324_s_s{s}", "ends_card_snap_nogeom_thr0.1.json"))
    rows = []
    for s in (0, 1):
        for lab, run, f in spec:
            c = J(f"runs/{run.format(s=s)}/{f}")
            if not c:
                rows.append([str(s), lab] + ["pending"] * 6)
                continue
            h = c["numbers"]["head"]
            rows.append([str(s), lab, f3(h["topo_F1"]), f3(h["topo_P"]), f3(h["junction_F1"]), f3(h["lanes_found"]), f3(h["dup_frac"]), f3(h["F1@1.0"])])
    return tbl(["seed", "read-out (p >= 0.1, VAL)", "topo F1", "topo precision", "junction F1", "lanes found", "dup_frac", "F1@1.0"], rows, num=(0, 2, 3, 4, 5, 6, 7))


def e320_table():''')
rep('<h3>E320: the learned attachment, read-outs on VAL</h3>', '<h3>E324 arm s: the E320 recipe at 6000 pair steps (the best read-outs so far)</h3>\n{e324_table()}\n<p class="cap">E324 s = geometry loss + learned attachment + 6000 pair steps (card E324, VAL, two seeds). The primary read-out (attached ends, the one-piece wiring on the output) beats the snapped geometry of E312 by +0.0087 / +0.0084 topo F1; the own-keypoint snapped read-out of the same run is +0.0156 / +0.0146 over E312 snapped with the guards held and is named a system-of-record CANDIDATE (reporting only; the adoption and any TEST use need Mirko\'s recorded OK). The duplicates guard of the primary read-out breaks again on seed 0 (+0.0150 against 0.01), so the written token is not KEEP; by D87 it is kept and refined (follow-up card E337: the attached end is a convex combination of the keypoint and raw B nodes, not a delivered piece end). Cost: +9.6 k parameters, +3.9 MMACs, no ms at the read-out; training 3.2 h per seed.</p>\n<h3>E320: the learned attachment, read-outs on VAL</h3>')
# best approaches: a row for E324 s
rep('    row("2", "E298, driven pairs weighted 2"', '    row("1b", "E324 s, snapped own keypoints (candidate, reporting)", "E320 recipe (geometry loss + attach) at 6000 pair steps; read out like E312 (snapped, p &gt;= 0.1)", cards("runs/e324_s_s{s}/ends_card_snap_thr0.1.json"), [None, None], pill("info", "as E312 + a learned attachment"), "the best read-out so far (+0.0156 / +0.0146 topo F1 over E312, guards held); adoption and any TEST need Mirko\'s recorded OK")\n    row("2", "E298, driven pairs weighted 2"')
open(p, "w", encoding="utf-8").write(s)
print("patched")
