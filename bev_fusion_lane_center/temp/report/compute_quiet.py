"""Report: the QUIET-host timing numbers into compute.json (lane lead's order of 2026-11-15 12:00): eager (c) and CUDA-graph (g1) ms per frame of E312 and E324 s from the four runs of E351
(`runs/eval/jtiming/e351_graph_20261115[b|c|d].json`, host load 0.8 - 1.2 on 12 cores, GPU idle, batch 1, bf16, 24 cached VAL frames, 10 rounds x 30 forwards), every earlier ms number marked as
'loaded host (E342 part B, load about 5.6)', and the price line of E324 s over E312 with its spread over the four sessions.
usage: python3 temp/report/compute_quiet.py   (plain python, no torch)  -> temp/report/compute.json (keys added: quiet_host, price_line_quiet; ms keys renamed with the loaded-host label)"""
import json
import os
import statistics

ROOT = os.environ.get("BFLC_ROOT") or os.getcwd()
P = os.path.join(ROOT, "temp", "report", "compute.json")
d = json.load(open(P))
tags = ["e351_graph_20261115", "e351_graph_20261115b", "e351_graph_20261115c", "e351_graph_20261115d"]
runs = [json.load(open(os.path.join(ROOT, "runs", "eval", "jtiming", t + ".json"))) for t in tags]
q = {"protocol": "runs/eval/jtiming/e351_graph_20261115[b|c|d].json (E351 card section 3): batch 1, bf16 autocast, 24 cached VAL frames, CUDA events, median of 10 block medians of 30 forwards, alternating blocks c / g1, cudnn.benchmark False; host 1-minute load 0.8 - 1.2 on 12 cores, GPU idle",
     "loads_before_after": [[r["load_before"], r["load_after"]] for r in runs], "systems": {}}
for run, label in (("e312_p_s0", "E312"), ("e324_s_s0", "E324 s (system of record)")):
    c = [r["systems"][run]["c_median_ms"] for r in runs]
    g = [r["systems"][run]["g1_median_ms"] for r in runs]
    pd = [r["systems"][run]["paired_g1_minus_c"][0] for r in runs]
    q["systems"][label] = dict(run=run, eager_ms_per_run=c, graph_ms_per_run=g, paired_graph_minus_eager_ms_per_run=pd, eager_median_ms=statistics.median(c), eager_min_max=[min(c), max(c)], graph_median_ms=statistics.median(g), graph_min_max=[min(g), max(g)],
                               paired_drop_median_ms=statistics.median(pd), percent_drop_median=statistics.median([r["systems"][run]["percent"] for r in runs]))
d["quiet_host"] = q
# every earlier number is a loaded-host number
for k, v in list(d.items()):
    if isinstance(v, dict) and "ms_per_frame" in v and k.startswith("B alone"):
        v["ms_per_frame_e273_timing_load_unknown"] = v.pop("ms_per_frame")
        v["ms_note"] = "LOAD NOT RECORDED (runs/e273 timing, an older measurement; not comparable with the quiet-host numbers): " + str(v.get("ms_note", ""))
    elif isinstance(v, dict) and "ms_per_frame" in v:
        v["ms_per_frame_loaded_host_e342b"] = v.pop("ms_per_frame")
        v["ms_note"] = "LOADED HOST (E342 part B, 1-minute load about 5.6 on 12 cores; the forward is host bound, so these numbers are inflated): " + str(v.get("ms_note", ""))
        if k in ("E312", "E324 s (system of record)"):
            v["ms_per_frame_quiet_eager"] = q["systems"][k]["eager_median_ms"]
            v["ms_per_frame_quiet_eager_min_max_over_4_sessions"] = q["systems"][k]["eager_min_max"]
            v["ms_per_frame_quiet_graph_g1"] = q["systems"][k]["graph_median_ms"]
        else:
            v["ms_per_frame_quiet_eager"] = "not measured on a quiet host yet (needs both GPU slots idle: first quiet window)"
# the price line of E324 s over E312 on the quiet host: topo F1 from the stored cards, ms per session
e312, e324 = q["systems"]["E312"], q["systems"]["E324 s (system of record)"]
dt = d["price_lines"]["E324 s (system of record) - E312"]["delta_topo_F1"]
dms = [b - a for a, b in zip(e312["eager_ms_per_run"], e324["eager_ms_per_run"])]
dg = [b - a for a, b in zip(e312["graph_ms_per_run"], e324["graph_ms_per_run"])]
d["price_line_quiet"] = dict(delta_topo_F1_mean_of_two_seeds=dt, eager_extra_ms_per_session=dms, eager_extra_ms_median=statistics.median(dms), eager_extra_ms_min_max=[min(dms), max(dms)],
                             graph_extra_ms_per_session=dg, graph_extra_ms_median=statistics.median(dg), graph_extra_ms_min_max=[min(dg), max(dg)],
                             topo_F1_per_ms_eager=dt / statistics.median(dms), topo_F1_per_ms_eager_range=[dt / max(dms), dt / min(dms)],
                             note="E324 s over E312: +topo F1 (snapped, mean of the two seeds, stored cards) for +ms per frame (quiet host, eager, seed-0 checkpoints, four sessions: the range is min - max over the sessions, not a bootstrap interval; the cross-system difference is not paired within a session)",
                             loaded_host_extra_ms_e342b=d["price_lines"]["E324 s (system of record) - E312"]["delta_ms"])
json.dump(d, open(P, "w"), indent=1)
print(json.dumps(d["quiet_host"]["systems"], indent=1)[:1500])
print(json.dumps(d["price_line_quiet"], indent=1))
