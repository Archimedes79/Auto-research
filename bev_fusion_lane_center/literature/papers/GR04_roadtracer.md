# GR04 - RoadTracer (fixed-step walker, dynamic oracle labels with map-matching)

*Pass: loop 16 (2026-10-23), cluster `growth` (iterative growth of road / lane graphs (agents, walkers, trackers)). Reading depth: **M+C** (A abstract, M method, C code page). Verified: checked (ar5iv text, README).. Pipeline stage: S4, T, L.*

## Papers

RoadTracer: Automatic Extraction of Road Networks from Aerial Images (Bastani et al., CVPR 2018) https://arxiv.org/abs/1802.03680

## Idea

A CNN walks along roads in fixed 12 m steps: per step an action (walk / stop) and a direction from a discrete angle set; branches are explored one after the other through a stack. The training is the lasting idea: the oracle labels the states that the CNN itself reaches (dynamic labels), because a static oracle trajectory made the CNN drift off the road after small angle errors.

## Formulation

Alg. 1: graph G and stack S start at v0; loop: CNN(G, S_top, window) -> (action, alpha); walk: u = S_top + D (cos a, sin a), D = 12 m, push u; stop: pop S. Merge rule: if u is within 3D (36 m) of an existing vertex v whose graph distance from S_top is at least 6D (72 m), add edge (u, v) and do not push (prevents small spurious loops). Outside the box = stop. Heads: 2-way softmax (walk / stop) + sigmoid over a angle bins (a not stated); walk if o_walk >= T (swept in evaluation), direction = argmax. Input window d x d (d not stated) of RGB + the current graph rendered as anti-aliased lines, 0.6 m / px. Dynamic labels: run the search with the CNN; at each step the oracle takes R = angles at S_top along which the GT has unexplored road: o*_stop = 1 if R empty, else o*_walk = 1 and a multi-hot over the nearest angle bins of all of R (all unexplored branches are targets, although only one is followed). Locating S_top in the GT: random walk of w = 10 vertices in G, map-matched to the GT with a Viterbi method; explored GT edges are remembered and avoided. Loss: cross-entropy on the action, MSE on the angle (static section; the dynamic section is not restated). Noise injection on static data: no meaningful gain.

## Input

RGB + graph channel window; `image_feat` + rendered history.

## Output

graph; per step one action and one direction (multi-hot target).

## Training

dynamic labels from CNN-driven states (`dagger`-like, no mixing with expert actions), GT graph + map-matching; no matching of sets.

## Differentiable?

no - argmax over angles, threshold T, stack control.

## Sensors and data

40 cities in 6 countries, 60 cm / px, OSM ground truth, 25 cities train / 15 test.

## Reported results

junction metric at 5 % average F_error: F_correct 0.58 vs 0.40 for a segmentation baseline (+45 % in the abstract); shortest-path agreement (SP) correct 0.72 vs 0.58 (segmentation) vs 0.21 (DeepRoadMapper, authors' re-implementation); TOPO better than segmentation for recall < 0.43 (Table 1 / figures).

## Compute

not stated.

## Code

https://github.com/mitroadmaps/roadtracer

## Fit to our project

reuse_idea - effort S. The multi-hot "all unexplored GT branches" target is a matching-free fixed-slot label for b children; map-matching plus an explored-edge memory handles revisits and off-GT states in free rollouts.

## Needs

GT graph with map-matching of arbitrary (off-road) states; an angle or sector discretisation.

## Conflicts

one branch followed per step at inference (no multi-hypothesis); fixed step; sequential stack.

## Tried in our experiments

none.
