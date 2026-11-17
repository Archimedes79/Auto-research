"""Append the night numbers (2026-11-16) to 03_numbers.json with their sources (scratch; run once)."""
import json, os
P = os.path.join(os.environ.get("BFLC_ROOT", os.getcwd()), "results", "Ergebnis_Spurmitte_2026-11-15", "parts", "03_numbers.json")
d = json.load(open(P, encoding="utf-8"))
d["numbers"] = [n for n in d["numbers"] if n.get("table") != "N"]
E354 = "experiments/E354_recursive-feature-state/experiment.md section 3"
E353 = "experiments/E353_paint-branch-width-25/experiment.md section 3"
E351 = "experiments/E351_launch-bound-junction-layer/experiment.md section 3b; runs/eval/jtiming/e351_quiet_eager.json"
E352 = "experiments/E352_pyramid-by-importance/experiment.md section 3"
E355 = "experiments/E355_one-token-set-design/experiment.md section 3b; experiments/E356_node-spacing/e355_step0_results.md"
E357 = "experiments/E357_oracle-nodes-one-pair-head/experiment.md section 3"
rows = [
    # E354
    ("E354 r minus z unsnapped topo_F1.s0", 0.0018, E354), ("E354 r minus z unsnapped topo_F1.s1", 0.0036, E354),
    ("E354 r minus z snapped topo_F1.s0", 0.0027, E354), ("E354 r minus z snapped topo_F1.s1", 0.0013, E354),
    ("E354 pair flip rate r minus z points.s0", -1.41, E354), ("E354 pair flip rate r minus z points.s1", 0.84, E354),
    ("E354 parameters added", 74368, E354), ("E354 GMACs added", 0.186, E354),
    ("E354 ms E341", 13.67, E354), ("E354 ms r", 14.20, E354), ("E354 ms r minus E341", 1.00, E354),
    # E353
    ("E353 snapped topo_F1 n1.s0", 0.6977, E353), ("E353 snapped topo_F1 n1.s1", 0.7008, E353),
    ("E353 n1 minus c snapped.s0", -0.0004, E353), ("E353 n1 minus c snapped.s1", -0.0002, E353),
    ("E353 GMACs c", 8.2016, E353), ("E353 GMACs n1", 7.0957, E353), ("E353 GMACs saved", 1.1059, E353),
    ("E353 parameters saved", 27664, E353), ("E353 ms E324 s", 19.951, E353), ("E353 ms n1", 19.957, E353),
    ("E353 ms n1 minus E324 s", -0.141, E353),
    # E351 quiet eager
    ("E351 q_eager ms E312", 15.13, E351), ("E351 q_eager ms E341", 14.29, E351),
    ("E351 q_eager ms E324 s", 20.58, E351), ("E351 q_eager ms E348", 19.86, E351),
    ("E351 q_eager E341 minus E312", -0.85, E351), ("E351 q_eager E348 minus E324 s", -0.80, E351),
    ("E351 q_eager E324 s minus E312", 5.45, E351),
    # E352
    ("E352 T1 GMACs", 0.2975, E352), ("E352 T2 GMACs", 0.8426, E352),
    ("E352 T2 B plain delta s0", -0.192, E352), ("E352 T2 B plain delta s1", -0.494, E352),
    # E355 step 0
    ("E355 sparse nodes per frame", 74.3, E355), ("E355 dense nodes per frame", 146.9, E355),
    ("E355 v25 junction_F1", 0.9855, E355), ("E355 s25 topo_F1", 0.9656, E355),
    ("E355 c25 topo_F1", 0.7744, E355),
    # E357
    ("E357 p topo_F1.s0", 0.4255, E357), ("E357 p topo_F1.s1", 0.3928, E357),
    ("E357 lo_near topo_F1", 0.7011, E357), ("E357 hi topo_F1", 0.9666, E357),
    ("E357 p junction-link F1.s0", 0.108, E357), ("E357 p junction-link F1.s1", 0.093, E357),
]
for name, v, src in rows:
    d["numbers"].append({"table": "N", "name": name, "value": v, "source": src})
d["written"] = "2026-11-15, night numbers (table N) added 2026-11-16"
d["count"] = len(d["numbers"])
json.dump(d, open(P, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("numbers", d["count"], "night rows", len(rows))
