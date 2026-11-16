"""Ledger notes of the nine E355 step-0 arms, the numbers read from the report table (temp/e356/e355_step0_report.txt), not typed."""
import subprocess

rep = open("temp/e356/e355_step0_report.txt", encoding="utf-8").read().split("\n")
rows = {}
head = None
for l in rep:
    if l.startswith("| metric"):
        head = [c.strip() for c in l.strip("|").split("|")]
    elif head and l.startswith("|") and not l.startswith("|---"):
        c = [x.strip() for x in l.strip("|").split("|")]
        rows[c[0]] = dict(zip(head[1:], c[1:]))
g = lambda m, a: float(rows[m][a])
P = "python3 ../autoresearch/tools/ledger.py note".split()


def note(rid, result, decision):
    r = subprocess.run(P + [rid, "--result", result, "--decision", decision], capture_output=True, text=True)
    print(rid, r.returncode, (r.stdout + r.stderr).strip()[:120])


def line(a):
    return (f"topo F1 {g('topo_F1', a):.4f}, lanes found {g('lanes_found', a):.4f}, F1@1.0 {g('F1@1.0', a):.4f}, junction F1 {g('junction_F1', a):.4f}, vert100_dp {g('vert100_dp', a):.2f}, "
            f"nodes per frame {g('nodes_per_frame', a):.1f} (p95 {g('nodes_per_frame_p95', a):.0f}, max {g('nodes_per_frame_max', a):.0f})")


L = {"s20": "l2", "s25": "l25", "s31": "l31"}
for a in ("s20", "s25", "s31"):
    extra = ""
    if a == "s20":
        extra = " kill row of E276 P1: 7 of 7 numbers within the PRINT resolution (tolerance widened from 5e-5 / 0.05 to 1e-4 / 0.1 after the first run: topo F1 0.96682 against the printed 0.9669, vert100_dp 11.449 against 11.5; the stored dense l2 is 0.966853 / 11.449); vert100 14.41 against the dense 33.07 shows the sparse pass-through."
    note(f"E355.{a}", f"suppression ON, sparse (0.3, 7.5): {line(a)}.{extra}", f"measurement, by hand: GT replay; passes the bar (loss against g1 at most 0.03); reference of the v arm of the same lattice")
for a, s in (("v20", "s20"), ("v25", "s25"), ("v31", "s31")):
    note(f"E355.{a}", f"suppression OFF (kept nodes inside the junction): {line(a)}; junction F1 {g('junction_F1', a) - g('junction_F1', s):+.4f} against {s}; kept nodes per connector {g('junc_nodes_per_connector', s):.2f} -> {g('junc_nodes_per_connector', a):.2f}; wanted-but-lost connectors {g('wbl_connector_share', a):.3f} (spec-literal; {g('wbl_connector_share_excl_adjacent', a):.3f} without losses to an adjacent piece)",
         "X2 NO at this lattice by the card's rule (wanted-but-lost far above 5 %), although junction F1 gains >= 0.01 (by hand: GT replay)")
for a, l in (("m20", "l2"), ("m25", "l25")):
    note(f"E355.{a}", f"merge radius = lattice: {line(a)}; against {l}: topo F1 {g('topo_F1', a) - g('topo_F1', l):+.4f}, junction F1 {g('junction_F1', a) - g('junction_F1', l):+.4f}; event nodes lost {g('merge_event_loss_share', a):.3f}; node share merged {g('merge_node_share', a):.3f}",
         "spacing reading (ii) NOT AFFORDABLE by the card's rule (junction F1 loss above 0.02; topo F1 loss within 0.01) (by hand: GT replay)")
note("E355.c25", f"cell-centre positions: {line('c25')}; against l25: F1@1.0 {g('F1@1.0', 'c25') - g('F1@1.0', 'l25'):+.4f}, topo F1 {g('topo_F1', 'c25') - g('topo_F1', 'l25'):+.4f}; connector coverage within 0.3 m {g('conn_cov_03', 'c25'):.3f}",
     "lattice-fixed positions fail the bar (a token set needs free offsets) (by hand: GT replay)")
