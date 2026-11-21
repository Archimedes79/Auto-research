# Models

The checkpoint files stay in `../runs/` (the live code loads them by those paths); this folder is the register. `REGISTRY_ALL.md` lists every checkpoint with size, date and hash (generated: `python tools/maintenance/make_structure.py`). The foundation is not trained here: GaussianCaR map-segmentation weights `~/work/gaussiancar_ckpt/nuscenes_map.ckpt` (WSL), code `~/work/gaussiancar`.

**`registry.json`** (in git) names the models that are evaluated and compared: `champion`, `champion_noprune`, `fast`, `fast_recall`, `recall`, `control`, each with checkpoint, config (`M`, `K0`), threshold and the standard assembly settings. `evaluation/evaluate.py <name>` uses exactly these; a new model is added there (with the ledger row that justifies it), not in a new evaluation script. The registry holds the models of the reference chain of loops 8 - 18 (its `champion` is the old route-graph champion, not the system of record); the token champion and the other token-system heads are not registry entries: their checkpoints are run folders (`e372_s4_s<S>`, `e373_cb_s<S>`, `e292_a_s<S>`, see the rows below) scored by `../evaluation/jtoken_eval.py`, and the best three networks are recorded in `../experiment_execution/keep3.json`.

| role | file (under `runs/`) | what | used by |
|---|---|---|---|
| **token champion** (system of record, D113 - D120) | `e373_cb_s<S>/` (stage-1 head), `e372_s4_s0/`, `e372_s4_s1/` (stage 2 = S4: the head and the node head of B' trained together; 2.323 M parameters) | VAL topo F1 0.6682 / 0.6813, 16.2 ms per frame (whole path, quiet GPU); no TEST check | `evaluation/jtoken_eval.py`, `training/train/jlive_train.py`; the source is `../../winners/e372_s4_champion` |
| lane head B' / plain B | `e292_a_s<S>/` (B': pyramid and node head, frozen under the token champion), `e273_p_s<S>/` (plain B, E273) | the front end of the token system, and its baseline (topo F1 0.6259 / 0.6249) | `training/grid/qgraph.py`, `training/data/lane_tokcache.py` |
| reference system E324 s | `e324_s_s<S>/` (pair head) with `e273_p_s<S>/` | VAL topo F1 0.6912 / 0.6948 at the end-to-end read-out, 19.7 ms; five rule steps | `training/lanegraph/` |
| route-association head (E389) | `e389_m_s<S>/` (with the SD route), `e389_z_s<S>/` (matched control without it) | TIE (no gain over the no-map head); not in the champion | `training/grid/jroute.py`, `evaluation/jroute_eval.py` |
| dense (grid) head | `e37_final/model.pt` (+ `prob_{val,test}.npy`, `metrics.json`) | U-Net on GaussianCaR features dec32 + logit7, 6,000 steps, heat map + direction | rule graph (system A), inputs of the loop-5/17 series |
| dense heads, out-of-fold | `e37_fold0/1/2/model.pt` | 3 folds for perturbed TRAIN rows | loop 17 failure fine-tune |
| trace network, **control** | `e94_a/model.pt` (+ `eval_*.json`, `graphs_*.pkl`) | E94 learned trace network (1.67 M) | control arm of loop 16 |
| refinement decoder | `e120_a/model.pt` ... `e120_d` | E120 refinement of the traced graph (3 layers) | control chain E94 + E120 |
| **champion builder** | `e123/e123_D7_r3_f1/model.pt` (4.7 MB) | D7 + junction head + time recursion (loop 16, rung 3) | `evaluate.py champion`, `p16_pipeline*.py`, `p16_fast_check.py` |
| builder, other designs | `e123/e123_D<n>_r<rung>_<step>/model.pt` | ladder runs D1-D8 | `experiment_execution/history/indexes/TRIAGE_STATE_loop16.md` |
| route head along the edges | `p16_f1m_route_s0/model.pt`, `..._s1`, `p16_f2h_route_s*`, `p16_rc1v_route_s*` | RouteXNet (cross-attention to SD-map tokens), loop 16b | `p10_route_train.py`, `p16_pipeline.py` |
| planner consumer | `e121_b/model_cons.pt`, `e117_*`, chain consumers `p16_cons_*` | ConsB (route + ego) | `p14_chain_consumer.py` |
| route head + planner loss | `e119_a/model_joint{0,1}_s{0,1}.pt` | joint training (no gain) | loop 14 |
| map-conditioned builder | `e123/e17_map_a`, `e17_map_b` (+ `e17_map_c`; `b2`, `c2` = crashed rerun, no model) | SD map raster as 5 extra input channels (loop 17) | `p17_mapnet.py` (its trainer `p17_mapft.py` was removed on 2026-11-21, the git history (commit `4c6ddf52`)) |
| learned gate | `e17/e134_gate/*_gate.pkl`, `e17/e140_gateB` | 12.5 k-parameter stabiliser (spatial memory gate) | `p17_gate.py` |

Rules: a checkpoint is registered here when it supports a number in a report; the run folder keeps `log.txt` and `status.json`. Do not overwrite a checkpoint - new runs get new folders (`CLAUDE.md`). Remove unreferenced checkpoints only via `_to_delete/`.

## The best three
`../experiment_execution/keep3.json` (made by `../../autoresearch/tools/keep3.py`) names the best three networks, one slot per checkpoint, ranked by Q on the full VAL split (card `../runs/eval/<name>/val.json`, band 0.010). Their checkpoints stay complete; every other checkpoint may be slimmed or archived, never deleted. `python3 ../../autoresearch/tools/keep3.py candidates` lists the networks, `consider <name> [--ledger <id> | --by-hand "reason"] [--apply]` takes one in. When the three change, commit the text records and push (rule: root `CLAUDE.md`, "Git").
