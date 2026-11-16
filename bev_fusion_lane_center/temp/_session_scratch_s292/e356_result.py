import io
import json
import subprocess
import time

p = "experiments/E356_node-spacing/experiment.md"
s = io.open(p, encoding="utf-8", newline="").read()
assert "## 3. Summary (2026" not in s
ta = subprocess.run(["python3", "temp/e356/e356_table.py", "A"], capture_output=True, text=True, check=True).stdout.split("\n\n")[0]
tb = subprocess.run(["python3", "temp/e356/e356_table.py", "B"], capture_output=True, text=True, check=True).stdout.split("\nl2 against")[0]
nn = json.load(open("temp/e356/nn_spacing.json"))
rows = ["| seed | read-out | nodes per frame (mean / p95 / max) | share with a neighbour < 2.0 m | share < 2.5 m | NN distance quartiles 25 / 50 / 75 % (m) |", "|---|---|---|---|---|---|"]
for k, v in nn.items():
    S, name = k.split("_", 1)
    q = v["nn_quantiles"]
    rows.append(f"| {S[1:]} | {name} | {v['nodes_per_frame']:.1f} / {v['nodes_p95']:.0f} / {v['nodes_max']} | {v['share_lt20']:.3f} | {v['share_lt25']:.3f} | {q[1]:.2f} / {q[2]:.2f} / {q[3]:.2f} |")
tn = "\n".join(rows)
m = {}
for n, path in (("l2", "temp/e356/out/e346_l2_full.json"), ("l25", "runs/eval/e346_l25/val.json"), ("l31", "runs/eval/e346_l31/val.json")):
    d = json.load(open(path))
    m[n] = d["metrics"] if "metrics" in d else d
cols = ("nodes_per_frame", "nodes_per_frame_p95", "nodes_per_frame_max", "outputs_per_frame", "nn_share_lt20", "nn_share_lt25", "nn_q50", "dropped_candidates_per_node", "coll_raw_share", "event_nodes_per_frame", "events_merged_per_frame", "readout_ms_mean")
tc = "| arm | nodes per frame (mean / p95 / max) | outputs per frame | share of decoded nodes with a neighbour < 2.0 m / < 2.5 m | NN median (m) | dropped candidates per node (E346 gate 1 %) | raw collision share of occupied cells | events per frame / merged | read-out ms |\n|---|---|---|---|---|---|---|---|---|\n"
for n in ("l2", "l25", "l31"):
    x = m[n]
    tc += f"| {n} | {x['nodes_per_frame']:.1f} / {x['nodes_per_frame_p95']:.0f} / {x['nodes_per_frame_max']:.0f} | {x['outputs_per_frame']:.0f} | {x['nn_share_lt20']:.3f} / {x['nn_share_lt25']:.3f} | {x['nn_q50']:.2f} | {100 * x['dropped_candidates_per_node']:.2f} % | {x['coll_raw_share']:.3f} | {x['event_nodes_per_frame']:.1f} / {x['events_merged_per_frame']:.2f} | {x['readout_ms_mean']:.2f} |\n"
now = time.strftime("%H:%M")
txt = f"""## 3. Summary (2026-11-15 {now}; CPU replays and GT replays, no training; by hand: replay of the cards with `temp/e356/e356_table.py`; chain `temp/day31/day31.sh`, run ids `E356.*`; VAL 873 frames, both seeds in part A)
**Result: (A) a post-hoc close-node radius of 2.0 or 2.5 m collapses B's graph and the system of record (the E275 result, reproduced to three decimals); (B) B's node + link code reproduces the lane GT on a 2.5 m and on a 3.125 m grid almost as well as on 2 m (both PASS the E346 bar; the loss against 2 m is 0.001 - 0.002 topo F1, the junction F1 falls by 0.007 / 0.017), but NEITHER grid gives a minimum spacing: 36 % (2.5 m) and 24 % (3.125 m) of the decoded GT nodes still have a neighbour closer than 2.0 m (92 % at 2 m).** D104's constraint ('no nodes closer than 2 m') is therefore not met by a coarser grid; it has to be a property of the node set itself (a spacing term, a merge or cell-centre positions: E355's arms m / c).
**(A) The read-out radius (the stored B dumps, `close:R@0.5`, B plain = `numbers.base`, E324 s snapped = `numbers.head`; vertices per 100 m after Douglas-Peucker next to every number):**

{ta}

The control recomputation (`E356.a15chk_snap_s0`, the tagged-score path with `--spec close:1.5@0.5 --tag c15`) reproduces the stored snapped card with a largest difference of 0.0 (kill row passed). **Reading, pre-registered: the E275 collapse is CONFIRMED** on all four (seed x numbers set) cells at both radii (every cell loses more than 0.10 topo F1): B plain topo F1 0.626 / 0.625 -> 0.238 / 0.220 at 2.0 m (E275: 0.238 / 0.220, identical) and 0.160 / 0.139 at 2.5 m; the E324 s snapped read-out 0.698 / 0.701 -> 0.404 / 0.399 and 0.324 / 0.312 (the junction connectors are snapped to a graph whose lanes are broken). The vertices per 100 m do NOT fall (B plain 19.3 -> 28.7 at 2.0 m: a broken graph has more short pieces), so neither radius is a candidate read-out; the delivered read-out stays `close:1.5@0.5`. **Why (measured on the stored B dumps, `temp/e356/nn_spacing.py`):**

{tn}

B's nodes sit at sub-cell positions, one per 2 m cell, so about 70 % of the delivered nodes have a neighbour closer than 2.0 m and the median nearest-neighbour distance is 1.99 m; radius 2.0 removes a third of the nodes (156 -> 106 per frame), 2.5 removes half (82), and B's short links, which join cells about 2 m apart, no longer find a partner.
**(B) The GT-coding ceiling on a coarser grid (E346 tool, arms l25 / l31 new, l2 = the stored E346 card; the decoded lane GT against the old graph GT; g1 = the undecoded lane GT):**

{tb}
**Pass rule (loss against g1 at most 0.03 on topo F1, lanes found, F1@1.0): l2 -0.0190 / -0.0146 / -0.0043 PASS; l25 -0.0202 / -0.0163 / -0.0047 PASS; l31 -0.0212 / -0.0209 / -0.0056 PASS.** The raw vertices per 100 m fall with the grid (33.1 -> 27.0 -> 21.8), the Douglas-Peucker vertices (the delivered number) do not change (11.45 / 11.41 / 11.38): a coarser grid is not a route to fewer delivered vertices. **Sizes and spacing of the decoded nodes:**

{tc}
Reading: the coarser grid saves nodes (-19 % / -35 % per frame; p95 and maximum 311 / 346 -> 254 / 280 -> 204 / 221, the sizes to dimension a token set) and outputs per frame (-36 % / -59 %) at 0.001 - 0.002 topo F1, but the share of nodes with a close neighbour falls only to 36 % / 24 % (below 2.0 m) and the median nearest-neighbour distance is the cell size minus a little (1.98 / 2.48 / 2.97 m): **no grid gives the guarantee**; the dropped candidates per node (E346's open item: 1.11 % against a gate of 1 %) are 1.11 % / 1.33 % / 2.61 %: the code at 3.125 m drops twice as many candidate nodes into occupied cells (the gate is exceeded on all three grids; the test of E346 stays red for the reason named there); events merged per frame rise 0.85 / 1.13 / 1.50 (an event place sharing its cell with another loses its own node).
**Predictions (written before; hit windows):** (A) B plain at 2.0 m 0.24 (window 0.20 to 0.30): HIT (0.238 / 0.220, both inside); at 2.5 m 0.20 (0.15 to 0.25): HIT on seed 0 (0.160), seed 1 just below the window (0.139); E324 s snapped loss at 2.0 m -0.35 (-0.45 to -0.25): HIT (-0.294 / -0.302); nodes per frame 106 / 82: HIT. (B) l25: topo F1 -0.025 against g1 (-0.032 to -0.020): HIT at the edge (-0.0202); lanes found -0.018 (-0.025 to -0.010): HIT (-0.0163); l31 topo F1 -0.030 (-0.040 to -0.022): MISSED on the good side (-0.0212); junction F1 predicted -0.065 / -0.085: smaller (-0.053 / -0.063); share of decoded nodes with a neighbour < 2.0 m: l2 0.70 MISSED (0.923), l25 0.30 (0.15 to 0.50) HIT (0.356), l31 0.10 (0.03 to 0.25) HIT at the edge (0.241).
**D106 line ('simpler AND faster?'):** this card changes no system. For the follow-up cards: a 3.125 m node code would have -35 % nodes, -59 % outputs and the same parts at the GT-code level; whether it is simpler AND faster than the 2 m code after training needs a trained arm (parts, parameters, GMACs of the link head, quiet-host ms): not claimed here.
**In-sample / caveats:** deterministic replays; the only noise is scene sampling (22 scenes). The 3.125 m grid is 4 % coarser than the 3 m of D104 (3.0 m does not divide the 100 m window); the l31 arm gets 5 long links on plain lanes (SHORT stays 8 m), named in the coder's notes (`temp/e356/build_notes.md`). `qnodes.py`, `qzones` and `qraster` assume N = 50: a trained head on these grids needs them changed (new files under the E343 freeze).
"""
a = "## 3. Summary\n(empty until the runs exist)"
assert s.count(a) == 1
s = s.replace(a, txt.rstrip("\n"))
io.open(p, "w", encoding="utf-8", newline="").write(s)
print("ok")
