# tools/prototype - the ready-to-use lane prototype (D28)

One command gives the delivered lane graph of a system on chosen frames, planner-ready, with its numbers, its ms per frame and (optionally) a results video. CPU only by default (cached network outputs); no GPU needed.

```
wsl bash tools/env/_py14.sh lane_prototype.py recall_dp --scenes 23 7 --video     # two VAL scenes (Boston, Singapore One-North) + video
wsl bash tools/env/_py14.sh lane_prototype.py recall_dp                           # the whole VAL split; must reproduce runs/eval/recall_dp/val.json
wsl bash tools/env/_py14.sh lane_prototype.py fast_recall_dp --scenes scene-0003 --frames 10 --out runs/prototype/try
```

| file | content |
|---|---|
| `lane_prototype.py` | the command: `<registry name>` (a builder or union entry of `models/registry.json`, default `recall_dp`; a union = the joined trees of its members, `evaluate.raw_of`), `--split val`, `--scenes <id or scene name> ...` (default all scenes of the split), `--frames N` (the first N frames of the selection, time order inside a scene), `--out <dir>` (default `runs/prototype/<name>/`, relative to the problem folder), `--video`, `--fresh` (re-run the network instead of the cache; CUDA only, refused without a GPU), `--test-id` (TEST only when logged in `evaluation/TEST_USAGE_LOG.md`) |
| `make_results_video.py` | results video (front camera + graph, prediction red over ground truth green, per-frame F1@1.0); `frame_table(split)` = the camera info of every frame, matched by timestamp; stand-alone it makes `reports/weekly/Lane_graph_<name>_val.mp4` from the two median scenes (moved from `temp/loop21/` on 2026-10-31) |
| `test_lane_prototype.py` | selftest (a few minutes, CPU): the coordinate rule (backward lanes on the left in Boston, on the right in Singapore), the JSON graph of two VAL scenes equals the graph `evaluate.py` scores and its numbers equal evaluate's on those frames, the whole VAL split reproduces the stored card |

## What it does
1. Graph: the cached raw network outputs of the registry entry (`evaluate.builder_raw`, the cache of `evaluate.py`), assembled with the registry's assembly settings and tau (`p16_infer.assemble`), then the registry's output stage `simplify_dp` (`dp_eps`) - the graphs `evaluate.py` scores (the selftest checks this frame by frame).
2. Numbers: `sparse_metrics.card_any(extra=True)` on exactly the chosen frames (the call of `evaluate.py`; no metric code here). Without `--scenes` / `--frames` the numbers must reproduce `runs/eval/<name>/<split>.json` within 5e-4 (layer 1; exit 1 otherwise).
3. ms per frame: read, never measured (no timing on a busy GPU): the latest stored session of `evaluation/drivers/p18_latency.py` (`runs/e18/latency/`) for this run with the registry's M, fork pruning and union; median of the replicate medians of `total` (builder, assembly, route input, route head, consumer; batch 1, cached features, idle GPU). The output stage `dp_eps` is not in the timed chain. If that session ran without the time recursion the system uses, the recursion cost measured in the same session on another run is given next to it (`ms_estimate_with_recursion`, once per builder). Sessions of the two-builder option (`--name2`) count only for a union entry with exactly those two members. summary.md shows two labelled estimates: the lane graph (builder + assembly of the same session, `lane_graph_ms`, no route head and no consumer) and the full chain, each with the time recursion when the system uses it.

## Output (`runs/prototype/<name>/`, not in git)

`scenes/<scene name>.json`, one per scene:

| field | meaning |
|---|---|
| `schema`, `system`, `registry`, `split` | format version 1, registry name and its entry (run, M / K0, tau, assembly settings) |
| `scene` | `id` (scene id of the poses / splits), `name` (nuScenes, e.g. `scene-0092`), `location`, `frames` |
| `frame` | the coordinate frame in words |
| `frames[]` | one entry per keyframe, in time order |
| `frames[].k`, `.sample_token`, `.timestamp_us` | frame position in the split, nuScenes sample token, timestamp in microseconds |
| `frames[].lanes[]` | the lane centre lines: `id` (index in this frame), `points` [[x, y], ...] in metres in the ego frame of the keyframe, **x forward, y left**, ordered in the **travel direction** (first point = where traffic comes from), `vertex_ids` (one per point; equal ids in two lanes = the same vertex, e.g. a fork or merge point), `confidence` (mean path validity `pi` of the network nodes the lane's vertices come from, 0..1; nodes below tau 0.7 are not in the graph, so values start near 0.7; `null` if no vertex has a source node) |
| `frames[].edges` | [[i, j], ...]: the end of lane i continues into lane j. A lane with two or more successors ends in a fork (split), a lane with two or more predecessors starts at a merge |
| `frames[].route` | `null`: the registry systems end at the lane graph; the route head of the latency chain is not part of this path |

summary.json: `system`, `registry`, `split`, `frames`, `scenes`, `fresh`, `source` (cache file), `metrics` (Q, topo_F1, lanes_found, R@1_far, vert100 = vertices per 100 m after the output stage), `metrics_all` (all of `card_any`), `assembly_ms_cpu` (assembly time of this call on the CPU: not a timing), `timing` (`ms`, `replicates`, `session`, `files`, `source`, recursion fields), `stored_card` (whole split only: `reproduced`, differences), `commit` (HEAD, no dirty check), `date`, `out`, `video`. summary.md says the same in a table.
