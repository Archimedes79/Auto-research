# E68: Bridging a perception outage: carrying the last good lane graph forward with the ego motion

Status: completed

**Question:** E67 keeps the ROUTE reference (the registered map route) usable through a sensor failure. A planner also needs LOCAL lane geometry - where the lanes are right now - and the perception delivers a nearly empty or wrong graph when the sensors fail (E57). Can the last good graph, folded into the current frame with the ego motion, stand in for it for a while? Which frames should it replace (the monitor of E64), how fast does it go stale in front of the vehicle, and what does an odometry error cost?

## Setup
- Training / test diagnosis: nothing is trained. The monitor is the E64 regression, cross-fitted over the parity of the frame index as in E67; the only free choice is the switch threshold p > 0.5 (the value E64 found sensible). Everything is TEST only; the clean frames before the failure are the seen-condition control, the frames after it the unseen ones.
- Scenario (as in E67): in each of the 21 TEST scenes the first half of the frames uses the graphs of the clean system, the second half the graphs of a perturbed run of E57 (night, three forward cameras off, blur, front camera off, fog, radar nearly off, extrinsics 3 deg); the failure lasts up to 10 s.
- Carried-forward graph: for a frame in the failure phase the SOURCE is the latest frame up to and including the current one that the monitor rates as good (p <= 0.5; oracle: graph F1 >= 0.6). Its extracted lane polylines are transformed into the current ego frame with the ego poses, p_cur = R_cur^T (t_src + R_src p_src - t_cur), and clipped to the +-48 m evaluation window; lanes that leave the window are cut. No topology is rebuilt (the metric is the point F1).
- Policies: perceived graph; carried-forward graph (source chosen by the monitor / by the oracle label); SWITCH = perceived graph while p <= 0.5, else the carried-forward one (monitor / oracle); UNION = perceived plus carried-forward when p > 0.5. Metric: point F1 @1 m of the whole +-48 m BEV (pooled over the frames: summed hits and counts), and for the part in front of the vehicle (y > 0) only, which is the part a planner needs; precision and recall; the F1 as a function of the age of the source frame. Ego motion: the dataset's ego poses (exact), and a variant with an odometry error (yaw N(0, 1 deg x sqrt(age in s)), offset N(0, 1 % of the distance travelled since the source frame)).

![architecture](E68_architecture.png)

## Numbers

**Failure phase: point F1 @1 m of the lane graph in the whole BEV (±48 m)**

| condition after the failure | frames (share true failures) | perceived graph (failing sensors) | carried-forward graph (monitor picks the source) | carried-forward graph (oracle picks the source) | switch: perceived while p <= 0.5, else carried forward | switch with oracle failure label | union of perceived and carried-forward when p > 0.5 |
|---|---|---|---|---|---|---|---|
| night | 420 (0.88) | 0.293 | 0.688 | 0.744 | 0.688 | 0.744 | 0.702 |
| 3 forward cameras off | 420 (0.31) | 0.666 | 0.668 | 0.714 | 0.668 | 0.714 | 0.671 |
| blur | 210 (0.68) | 0.474 | 0.697 | 0.737 | 0.697 | 0.737 | 0.711 |
| front camera off | 420 (0.14) | 0.766 | 0.773 | 0.781 | 0.772 | 0.781 | 0.773 |
| fog | 420 (0.09) | 0.830 | 0.834 | 0.842 | 0.834 | 0.842 | 0.837 |
| radar nearly off | 210 (0.10) | 0.829 | 0.831 | 0.845 | 0.831 | 0.845 | 0.831 |
| extrinsics 3 deg | 210 (0.25) | 0.700 | 0.700 | 0.742 | 0.700 | 0.742 | 0.706 |
| mean over the conditions | - | 0.651 | 0.741 | 0.772 | 0.741 | 0.772 | 0.747 |

**Failure phase: point F1 @1 m in front of the vehicle only (y > 0), the part a planner needs**

| condition after the failure | frames (share true failures) | perceived graph (failing sensors) | carried-forward graph (monitor picks the source) | carried-forward graph (oracle picks the source) | switch: perceived while p <= 0.5, else carried forward | switch with oracle failure label | union of perceived and carried-forward when p > 0.5 |
|---|---|---|---|---|---|---|---|
| night | 420 (0.88) | 0.318 | 0.578 | 0.637 | 0.578 | 0.637 | 0.616 |
| 3 forward cameras off | 420 (0.31) | 0.507 | 0.500 | 0.562 | 0.499 | 0.561 | 0.506 |
| blur | 210 (0.68) | 0.462 | 0.602 | 0.630 | 0.602 | 0.630 | 0.637 |
| front camera off | 420 (0.14) | 0.697 | 0.697 | 0.700 | 0.697 | 0.700 | 0.701 |
| fog | 420 (0.09) | 0.848 | 0.841 | 0.848 | 0.842 | 0.848 | 0.850 |
| radar nearly off | 210 (0.10) | 0.863 | 0.858 | 0.865 | 0.857 | 0.865 | 0.861 |
| extrinsics 3 deg | 210 (0.25) | 0.710 | 0.704 | 0.734 | 0.704 | 0.734 | 0.712 |
| mean over the conditions | - | 0.629 | 0.683 | 0.711 | 0.683 | 0.711 | 0.698 |

**Precision and recall (failure phase, whole BEV) and the cost of the monitor's false alarms before the failure**

| condition | perceived: precision / recall | switch (monitor): precision / recall | union (monitor): precision / recall | clean frames before the failure: perceived F1 | switch (monitor) F1 (false alarms of the monitor) |
|---|---|---|---|---|---|
| night | 0.71 / 0.18 | 0.87 / 0.57 | 0.83 / 0.61 | 0.864 | 0.864 |
| 3 forward cameras off | 0.74 / 0.61 | 0.75 / 0.60 | 0.72 / 0.63 | 0.864 | 0.864 |
| blur | 0.77 / 0.34 | 0.87 / 0.58 | 0.82 / 0.63 | 0.862 | 0.862 |
| front camera off | 0.82 / 0.72 | 0.84 / 0.72 | 0.81 / 0.74 | 0.864 | 0.864 |
| fog | 0.87 / 0.79 | 0.88 / 0.79 | 0.86 / 0.81 | 0.864 | 0.864 |
| radar nearly off | 0.85 / 0.80 | 0.86 / 0.80 | 0.84 / 0.82 | 0.862 | 0.862 |
| extrinsics 3 deg | 0.73 / 0.68 | 0.74 / 0.66 | 0.72 / 0.69 | 0.862 | 0.862 |

**F1 by the age of the source frame (monitor-flagged frames of all conditions, pooled counts)**

| age of the source frame | frames flagged by the monitor | perceived graph (same frames): F1 / in front | carried forward (monitor picks the source): F1 / in front | carried forward (oracle picks the source): F1 / in front |
|---|---|---|---|---|
| < 1 s | 95 / 98 (oracle) | 0.462 / 0.401 | 0.654 / 0.577 | 0.732 / 0.682 |
| 1-2 s | 132 / 136 (oracle) | 0.421 / 0.398 | 0.681 / 0.604 | 0.751 / 0.720 |
| 2-4 s | 167 / 195 (oracle) | 0.387 / 0.350 | 0.683 / 0.566 | 0.730 / 0.637 |
| 4-7 s | 159 / 212 (oracle) | 0.335 / 0.319 | 0.660 / 0.489 | 0.702 / 0.531 |
| >= 7 s | 185 / 172 (oracle) | 0.288 / 0.286 | 0.540 / 0.292 | 0.615 / 0.352 |

**Ego-motion error: the same policies when the odometry has an error (yaw 1 deg per sqrt(s) and 1 % of the travelled distance)**

| policy (mean over the conditions) | failure phase: exact ego motion | failure phase: ego motion with error (1 deg / sqrt(s) yaw, 1 % of the distance) | failure phase, in front of the vehicle: exact | with error | exact ego motion, monitor fitted on other scenes (scene-disjoint) |
|---|---|---|---|---|---|
| perceived graph (failing sensors) | 0.651 | 0.651 | 0.629 | 0.629 | 0.651 |
| carried-forward graph (monitor picks the source) | 0.741 | 0.711 | 0.683 | 0.657 | 0.736 |
| carried-forward graph (oracle picks the source) | 0.772 | 0.732 | 0.711 | 0.678 | 0.772 |
| switch: perceived while p <= 0.5, else carried forward | 0.741 | 0.710 | 0.683 | 0.657 | 0.736 |
| switch with oracle failure label | 0.772 | 0.732 | 0.711 | 0.678 | 0.772 |
| union of perceived and carried-forward when p > 0.5 | 0.747 | 0.724 | 0.698 | 0.679 | 0.744 |

## Result images

![Point F1 @1 m of the lane graph after the sensor failure began, per condition and policy (left: whole BEV, right: only the part in front of the vehicle).](E68_e68_bars.png)
*Point F1 @1 m of the lane graph after the sensor failure began, per condition and policy (left: whole BEV, right: only the part in front of the vehicle).*

![F1 of the carried-forward graph as a function of the age of its source frame (frames flagged by the monitor, all conditions pooled): the perceived graph of the same frames is the horizontal reference.](E68_e68_age.png)
*F1 of the carried-forward graph as a function of the age of its source frame (frames flagged by the monitor, all conditions pooled): the perceived graph of the same frames is the horizontal reference.*

![Night, one scene, 1.5 s / 5 s / 9 s after the failure began (x right, y forward, ego at the origin): ground truth (green), the nearly empty perceived graph (black) and the graph carried forward from the last good frame (orange). The carried-forward graph keeps the lanes behind and beside the vehicle; the lanes ahead of it end where the old frame ended.](E68_e68_bev.png)
*Night, one scene, 1.5 s / 5 s / 9 s after the failure began (x right, y forward, ego at the origin): ground truth (green), the nearly empty perceived graph (black) and the graph carried forward from the last good frame (orange). The carried-forward graph keeps the lanes behind and beside the vehicle; the lanes ahead of it end where the old frame ended.*

## Findings
- Carrying the last good graph forward lifts a failing lane graph substantially. Mean over the seven failure types, failure phase, whole BEV: point F1 0.651 for the perceived graph, 0.741 for the monitor-gated switch, 0.772 if an oracle chose the source frames, 0.747 for the union. The gain sits where the perception truly fails - night 0.293 -> 0.688, blur 0.474 -> 0.697 - and is nil where it still delivers (front camera off 0.766 -> 0.772, fog 0.830 -> 0.834, radar nearly off 0.829 -> 0.831, three forward cameras off 0.666 -> 0.668, extrinsics 0.700 -> 0.700). Before the failure the switch costs nothing (0.863 for both): a false alarm of the monitor swaps a good frame for one a fraction of a second older. With a monitor fitted on other scenes than the ones it judges (scene-disjoint cross-fit, E64) the switch reaches 0.736 instead of 0.741 (night 0.674 instead of 0.688; clean frames 0.860 against 0.863).
- What the failing perception loses is RECALL, and the carried-forward graph returns it without paying in precision: at night precision / recall 0.71 / 0.18 -> 0.87 / 0.57, under blur 0.77 / 0.34 -> 0.87 / 0.58. The union adds a little recall (0.61 / 0.63) for a small loss of precision (0.83 / 0.82); in front of the vehicle it is the best variant at night and under blur (0.616 / 0.637 against 0.578 / 0.602 for the switch), but over all conditions the switch and the union differ by 0.006 (whole BEV) and 0.015 (in front) - the choice is about whether false lanes or missing lanes cost more, and for this project false alarms cost more (the switch).
- The carried-forward graph is a decaying picture. F1 by the age of the source frame (frames the monitor flags, all conditions pooled): 0.654 / 0.681 / 0.683 / 0.660 / 0.540 for < 1 s / 1-2 / 2-4 / 4-7 / >= 7 s against 0.46 -> 0.29 for the perceived graph of the same frames. In FRONT of the vehicle it decays faster: 0.577 / 0.604 / 0.566 / 0.489 / 0.292 - from about 7 s on the carried-forward lanes ahead are no better than the failing perception (0.292 against 0.286 at >= 7 s; 0.489 against 0.319 at 4-7 s), because the old frame ended where the vehicle now is, while the lanes behind and beside it stay valid (the figure shows a moving vehicle at night: 0.88 at 1.9 s, 0.70 at 5.4 s, 0.36 at 9.4 s). With an oracle choice of the source frame the curve is higher (0.73 -> 0.62 whole, 0.68 -> 0.35 in front), so a better source selection could add about 0.04-0.08 (0.04-0.12 in front), but not remove the decay. Rule of thumb: the carried-forward graph is worth about 5 s of driving; beyond that the planner has to rely on the map (E67).
- The ego-motion error costs more the older the source: with an odometry error of 1 deg per sqrt(s) in yaw and 1 % of the distance, the mean F1 of the monitor-gated switch falls from 0.741 to 0.710 (in front 0.683 -> 0.657); by age bin the whole-BEV F1 loses 0.03 / 0.06 / 0.09 / 0.15 / 0.15 (< 1 s ... >= 7 s; 0.628 / 0.625 / 0.598 / 0.510 / 0.389), and in front of the vehicle the carried-forward graph is then as good as the failing perception at 4-7 s (0.344 against 0.319) and worse after 7 s (0.207 against 0.286). The whole-BEV F1 stays above the perceived one throughout, because the old lanes behind and beside the vehicle remain right.
- Reading for the system: the layers that cover a failure have different horizons. (1) The registered map route, held through the failure (E67), is the primary route reference for as long as the localisation error stays slow (about 10 s tested). (2) For the lane geometry, a current map placed by the same held registration is the better source (E69: F1 0.89 against 0.74 for the carried-forward graph, circular on nuScenes); the carried-forward lane graph is the map-free alternative and an independent cross-check, worth about 5 s of driving. (3) When neither is trusted, the planner should be told that the lane perception is gone (the monitor output) so that it can slow down or hand over. The carried-forward graph is cheap (a rigid transform of the last good polylines) and uses nothing that is not already in the system: the ego poses, the monitor and the last good graph.

## Limits
- The ego poses of nuScenes are exact; the odometry-error variant is a synthetic model (yaw error growing with the square root of the age, 1 % distance error), not a recorded odometry. Real odometry fails differently on a slippery road or in a turn.
- Static lanes only: the carried-forward graph assumes the lane geometry did not change; it cannot show lanes that were outside the +-48 m window of the source frame, which is why the part in front of a moving vehicle goes stale. No topology (successor edges, splits / merges) is rebuilt and the metric is the point F1 @1 m; the route metrics of the lane graph are not evaluated here because the map route (E67) is the primary route.
- The failures are the synthetic corruptions of E57 starting abruptly in the middle of every scene and lasting up to 10 s (20 or 10 frames); real failures are gradual, intermittent and of any length, and a failure that begins together with the drive has no good source frame at all (the system then falls back on the perceived graph; the first frames of a scene are not covered by the test).
- The source frame is chosen with the monitor's failure probability, which is judged on the same synthetic perturbations (E64: held-out AUROC 0.95 on the frames of the same conditions); a monitor false negative would carry a bad graph forward. The oracle rows show the ceiling of a perfect choice.
- Single scenario per condition on 21 TEST scenes, one run (the policy has no random part except the odometry noise); frames of a scene are strongly correlated.
