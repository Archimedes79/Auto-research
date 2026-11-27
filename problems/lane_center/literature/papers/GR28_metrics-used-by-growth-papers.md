# GR28 - Metrics used by growth papers (TOPO, APLS, junction metric, SDA, landmark / reachability, StreetMover, Betti)

*Pass: loop 16 (2026-10-23), cluster `growth` (iterative growth of road / lane graphs (agents, walkers, trackers)). Reading depth: **M** (A abstract, M method, C code page). Verified: . Pipeline stage: E - idea (definitions as read): Junction metric (RoadTracer): a junction = vertex with >= 3 edges; each GT junction matched to an inferred one; f_correct = share of GT edges captured around it, f_error = share of inferred edges not in the GT. TOPO: simulate driving from seed locations for a fixed distance, compare reachable destinations. SP: fraction of sampled origin-destination pairs whose shortest paths agree within 5 %. Landmark P/R (RNTR): nearest-landmark match within 0.5 - 5 m thresholds; reachability P/R: path Chamfer distance < 0.5 - 2.5 m for matched landmark pairs within 5 edges. SDA20 / SDA50 (LaneGNN): split detection accuracy within 20 / 50 px. Point-level F1 with one-to-one matching within 1.5 voxels and branch-F1 (>= 80 % coverage), Betti-0 / Betti-1 errors (Trexplorer Super) avoid linear-assignment metrics. StreetMover: Sinkhorn-Wasserstein between sampled point clouds..*

## Papers

RoadTracer (junction metric, TOPO, SP); RNTR / SeqGrowGraph (landmark and reachability); LaneGNN (GEO / TOPO P-R, SDA20 / SDA50, graph IoU); Trexplorer Super (point, branch, Betti); Belli and Kipf (StreetMover); TopoGPT (M-P / M-R / M-T, G-IoU, SDA, G-F1, T-F1, JT-F1, APLS); links in the cards above.

## Fit to our project

reuse_idea - effort S: for fork / merge evaluation use landmark P/R for out-degree-2 / in-degree-2 vertices (cf. lit_p16_tree, item 12) and report SDA; our split / merge F1 at 6 m already is a landmark metric.

## Tried in our experiments

split / merge F1 at 6 m (project metric). verified: checked.
