# VM11 - SparseDrive map branch (temporal map instances next to a planner)

*Pass: loop 16 (2026-10-23), cluster `vectormap_temporal` (online vector maps and temporal / streaming fusion). Reading depth: **M** (A abstract, M method, C code page). Verified: partly.. Pipeline stage: S8, S10.*

## Papers

SparseDrive: End-to-End Autonomous Driving via Sparse Scene Representation (Sun et al., ICRA 2025) https://arxiv.org/abs/2405.19620

## Idea

Map polylines are sparse instance anchors (N_m = 100, 20 points each) handled by the same instance-bank mechanism as detected agents (temporal decoders cross-attend last-frame instances to the current ones; the page reports 33 temporal map instances); a parallel motion planner reads agents and map with agent-temporal / agent-agent / agent-map attention and an ego-instance memory queue of 3 frames.

## Formulation

L_map = 1 x focal + 10 x L1 with Hungarian matching; permutation / direction handling for polylines not described; top-k selection and ego transform of map anchors not described on the page. Stage 1 perception from scratch (S 100 epochs, B 80), stage 2 joint with the planner (10 epochs).

## Input

images -> sparse map and agent instances.

## Output

map polylines, agents, ego trajectory.

## Training

hungarian_set.

## Differentiable?

partial.

## Sensors and data

camera; nuScenes, map range 60 x 30 m, AP thresholds not stated.

## Reported results

nuScenes val map mAP: B 56.2 (ped 53.2, divider 56.3, boundary 59.1), S 55.1 (VAD 47.6); planning: B L2 0.58 m, collision 0.06 %; S 0.61 m, 0.08 %.

## Compute

not stated here.

## Code

https://github.com/swc-17/SparseDrive

## Fit to our project

reuse_idea - S - confirms that published stacks give map instances the same temporal instance memory as agents and hand them to the planner by attention; the planner-side interface (S10) is what E117 already does with edges, route probability and ego motion.

## Needs

instance map output with confidences.

## Conflicts

Hungarian; planner trained jointly end to end (ours: planner loss through positions gave no gain).

## Tried in our experiments

E117 planner consumer (0.34 m at 20 m), E119 / E121 planner loss into route head / positions (no gain).
