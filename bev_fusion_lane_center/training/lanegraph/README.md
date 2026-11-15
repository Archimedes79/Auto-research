# lanegraph: the champion code of the lane-graph system of record (E324 s, D94)

A clean re-implementation of the inference path of the system of record: camera + radar BEV features in, a lane-centre-line graph out
(lane pieces with their successors, connectors through junctions). It loads the trained checkpoints unchanged and gives **bit-identical** outputs
to the original modules (`training/grid`, `evaluation/`) on CPU. Training code is not part of it. The modules are imported flat (the project's
convention) and carry the prefix `lg_` so that no name collides with another module on the project path.

## Architecture
```
x (39 x 200 x 200 cached GaussianCaR BEV features, 0.5 m)                            lg_features.py
 └─ Pyramid: encoder-decoder, maps at 2 / 1 / 0.5 m                                    lg_pyramid.py
     ├─ B, the lane head (E273)                                                        lg_lane_head.py
     │    NodeHead: a node per 2 m cell (existence, type flags, position, heading; 7 x 7 neighbourhood attention + refinement)
     │    LinkHead: top-512 nodes, short / long candidate links, set attention, link logit + handle scales
     │    section membership: soft absorbing walk over the links
     └─ junction head
          JunctionMaps (E307): entry / exit heat + heading at 1 m, zone, stop lines    lg_junction_maps.py
          keypoints: 32 entry + 32 exit heat peaks, soft-argmax position              lg_junction_maps.py
          PairHead (E324 s): set attention over the keypoints; geometry, evidence along the curve and the
                             order rule (rank in a group of parallel lane ends) -> pair logit, handle scales   lg_pair_head.py
          Attach (E320): learned soft attachment of the connector ends to B's nodes   lg_attach.py
 └─ read-out = the NAMED HARD STEPS (numpy)                                             lg_readout.py
      0 section = argmax of the membership | 1 close-node suppression 1.5 m @ e >= 0.5 | 2 links p >= 0.5 | 3 section chains |
      4 pairs p >= 0.1 (allowed only) | 5 drop B's long links a connector covers | 6 wire each connector end to the nearest piece end (4 m, 45 deg)
 └─ Graph (polylines, edges, node ids, kinds) -> JSON with Douglas-Peucker 0.3 m       lg_graph.py
```
`lg_model.py` wires it and holds the checkpoint key map (`B_KEYS`, `J_KEYS`) and the checked training options; `lg_config.py` holds every selection
and read-out setting (the fixed input scalings of the trained layers are part of the weights and stay in the layer code). The model hands the
read-out one `LaneFrame` and one `PairFrame` per frame (`lg_readout.py`, TypedDicts). Everything before the read-out is soft and differentiable
(index choices: the candidate sets).

**Two read-outs.** The card of record (`attached_ends=False`) draws each connector between the two wired piece ends (the snap of E311: a rule step,
the named E2E gap): VAL topo F1 0.6981 / 0.7010 (seed 0 / 1, `runs/e324_s_s<S>/ends_card_snap_thr0.1.json`). `attached_ends=True` draws it between
the learned attached ends and still wires it by the same 4 m / 45 degree nearest-piece rule (one piece per end): 0.6912 / 0.6948
(the card ends_card_snap_nogeom_attach_thr0.1 of that run folder, the 'primary' read-out of E324). Neither is rule-free; the attached ends without that wiring rule
(every piece in reach, the card ends_card_attach_thr0.1) score 0.6616 / 0.6646 and are not implemented here.

## Run (from `bev_fusion_lane_center/`; `_py14.sh` changes into the script's folder, so give `--out` as an absolute path)
```
bash tools/env/_py14.sh training/lanegraph/scripts/lg_infer.py --seed 0 --first 20 --out /abs/path/graphs.json [--device cuda]   # VAL only
bash tools/env/_py14.sh training/lanegraph/scripts/lg_eval.py --seed 0 --frames 873 [--device cuda]     # project scorer, both read-outs, vs the stored cards
bash tools/env/_py14.sh training/lanegraph/scripts/lg_timing.py --seed 0 [--device cuda]                # ms per frame next to the original modules
bash tools/env/_py14.sh training/lanegraph/tests/run_lg_tests.py [--quick]                              # tests (below)
```
```python
from lg_model import LaneGraphNet; from lg_features import FeatureStore, split_rows
net = LaneGraphNet.from_runs(seed=0)                          # runs/e273_p_s0/model.pt (B) + runs/e324_s_s0/model.pt; eval mode; raises on other options
graphs = net.graphs(FeatureStore()(split_rows("val")[:4]))
```
Weights stay in `runs/`. A float16 copy of seed 0 is `results/Ergebnis_Spurmitte_2026-11-15/champion_weights_fp16.pt` (5.8 MB, a dict
`{"state": LaneGraphNet state dict in float16, "note": ...}`; load with `{k: v.float()}`; float16 weights move the pair logits by up to 0.1).

## What is reproduced (VAL)
- **Equivalence, exact** (`run_lg_tests.py --seeds 0 1 --frames 96 48 --stride 9`): every stage (pyramid, nodes, links, sections, maps,
  keypoints, pair logits, scales, attached ends: 41 quantities) and the delivered graphs of both read-outs (points, edges, node ids, kinds) are
  identical to the original modules, CPU float32, on 96 (seed 0) and 48 (seed 1) frames spread over VAL; Douglas-Peucker equals the project's
  `sparse_metrics._dp_keep`; the read-out alone equals the original read-out on 600 synthetic graphs (2,604 connectors, 580 wired ends, 11 covered
  long links); golden digests (structure hash + coordinate sum) of three graphs. Compared on the 96 + 48 frames: 4,683 lane, 2,070 link and
  1,991 connector pieces. The default run (`run_lg_tests.py`: 24 frames per seed, every 36th row) takes about 2 minutes.
- **Batch caveat.** The original LinkHead subsamples at random when a batch holds more than 80,000 link candidates (`qlinks.py`); the champion has no
  cap. Bit identity therefore holds while a batch of the original stays under it. Measured on all 873 VAL frames: 12.3 k candidates per frame;
  in the batches of 4 of B's stored dump (`qgraph_run_eval.predict_val`, bs 4) at most 69,029 (seed 0) / 67,785 (seed 1): the cap never fired for
  the cards; in batches of 8 it would fire in 84 of 110 batches (the junction dumps at bs 8 run no link head).
- **Card numbers** (`lg_eval.py --frames 873 --device cuda`, run.py diagnostics `lanegraph.eval873_s0 / _s1`), topo F1 champion (stored card):
  snapped 0.6982 (0.6981) / 0.7010 (0.7010), attached ends 0.6912 (0.6912) / 0.6948 (0.6948); every headline number within 0.0007 of the
  card. Frames with identical per-frame counts: snapped 726 / 742 of 873, attached 591 / 543. That the rest differ is probably the floating-point
  difference between this run and the GPU runs that made the stored dumps (other process, batch and kernels; on CPU the champion equals the original
  modules exactly); it is probable, not proven.
- **Speed** (`lg_timing.py`, batch 1, float32, loaded host): GPU (idle, RTX 3060, 12 frames x 8 alternating rounds) champion **42.3 ms** per
  frame against 60.0 ms for the original modules as the evaluation runs them (B's QGraph to its sections + the junction model, which computes the
  pyramid and the node head a second time; mean paired difference -20.1 ms, range -30.1 .. -13.3), with the attachment 43.5 ms; CPU (2 threads)
  370 / 583 / 390 ms. The numpy read-out takes about 23 ms per frame. The saving is the second pyramid and node-head pass, not faster layers.
  (The card E342's 24.6 ms is the junction model alone in bf16.)

## Size
2.86 M parameters (pyramid 2.10 M, nodes 0.10, links 0.12, junction maps 0.44, pair head 0.09, attachment 0.01). The 11 modules have 913
lines, none over 170; scripts 233, tests 278. The original path spans 4,409 lines in 14 files (training, options and checks included).

## Limitations
- No training code and no losses (dual softmax, crossing penalty, geometry loss, attachment cross entropy): `training/train/qgraph_train.py`,
  `training/train/jmaps_train.py`, `training/train/jends_train.py`, `training/grid/jends.py`. Recipe: B = E273 p; maps E307 n (3000 steps); pair head 6000 steps
  `--e2e --scales --maps-w 1.0 --geom-w 0.5 --attach` on the frozen pyramid.
- float32 only (training used bf16 autocast). Nothing on TEST.
- The evidence of the link head is sampled frame by frame (one host sync per frame).
