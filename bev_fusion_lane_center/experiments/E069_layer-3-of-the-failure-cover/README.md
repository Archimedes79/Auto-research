# E69: Layer 3 of the failure cover: the registered map lanes as the lane graph when the perception is gone

Status: completed

**Question:** E67 keeps the ROUTE reference through a failure and E68 carries the last good LANE GRAPH forward for a few seconds. When the map is current, the map itself contains the lane geometry: placed by the held registration of E67 it could serve as the lane graph for as long as the outage lasts. How good is it at the 1 m tolerance of the lane-graph metric, against the failing perception and the carried-forward graph, how does it behave over time, and what do an outdated map (30 % of the lanes removed or re-painted) and a looser tolerance change? A caveat is built in and stated first: on nuScenes the ground-truth lane graph IS the map, so the experiment is circular - it measures what the registration error and the staleness of the map cost, not the benefit of an independent map.

## Setup
- Training / test diagnosis: nothing is trained. The registrations are those of E67 (main setting: scene-constant map pose error of 1 m / 2 deg that grows with the distance driven, jitter 0.2 m / 0.3 deg; monitor gate p > 0.5, cross-fitted over the frame parity) on the same 21 TEST scenes and the same failure scenario (second half of every scene perturbed, up to 10 s); the carried-forward graph and the perceived graph are those of E68. Everything is TEST only; before the failure the system is in its clean state and is not part of the numbers.
- Map lanes as the lane graph: the BIASED map (the same disturbed lane polylines that E67 registers) is transformed into the ego frame with the registration of a method and treated as the lane graph of the frame: (1) as given (no registration), (2) per-frame graph-ICP registration, (3) tracked registration, (4) the registration HELD through the failure (monitor gate p > 0.5), (5) the oracle gate (true failure label), (6) the TRUE registration, which is the ceiling and equals 1 by construction because the ground truth is the map.
- Source selection (policy_eval.py): in the failure phase the lane geometry can come from three sources - the perceived graph (used while the monitor says p <= 0.5), the carried-forward graph (E68) and the map lanes under the held registration (E69; current map, 30 % removed, 30 % shifted, or a WRONG map = the map of a random sample of another scene shown in this frame). Rule A takes the map lanes if the carried-forward graph (source frame at most 3 s old) agrees with them, otherwise the carried-forward graph; rule B takes them while a MAP-VALIDITY signal is high - the agreement between the perceived graph of the last ten healthy frames and the registered map lanes - otherwise the carried-forward graph. The thresholds are the 5 % quantiles of the signals on the clean frames of the current map (no labels).
- Outdated map (applied to the lanes before the pose error; the registrations are those computed with the current map): 30 % of the lanes removed, or 30 % of the lanes shifted sideways by 1.5 m (re-painted). Metric: point F1 @1 m against the ground-truth lane graph of the frame, pooled over the failure-phase frames, for the whole +-48 m BEV and for the part in front of the vehicle (y > 0), by the time since the failure began; the sensitivity to the tolerance is shown with 1.5 m.

![architecture](architecture.png)

## Numbers

**Failure phase: point F1 @1 m of the lane graph in the whole BEV (+-48 m). Map lanes = the biased map transformed into the ego frame with the registration of E67; the ground truth IS the map, so the true-registration column is the ceiling (1.0)**

| condition after the failure | perceived graph (E68) | last good graph carried forward (E68, monitor switch) | map lanes as given (no registration) | map lanes, per-frame registration | map lanes, tracked registration (every frame) | map lanes, held registration (monitor gate) | map lanes, oracle gate | map lanes, TRUE registration (ceiling: 1 by construction) |
|---|---|---|---|---|---|---|---|---|
| night | 0.293 | 0.688 | 0.568 | 0.579 | 0.677 | 0.843 | 0.873 | 1.000 |
| 3 forward cameras off | 0.666 | 0.668 | 0.568 | 0.803 | 0.878 | 0.894 | 0.920 | 1.000 |
| blur | 0.474 | 0.697 | 0.565 | 0.654 | 0.742 | 0.895 | 0.915 | 1.000 |
| front camera off | 0.766 | 0.772 | 0.568 | 0.843 | 0.901 | 0.906 | 0.918 | 1.000 |
| fog | 0.830 | 0.834 | 0.568 | 0.879 | 0.948 | 0.951 | 0.958 | 1.000 |
| radar nearly off | 0.829 | 0.831 | 0.565 | 0.858 | 0.929 | 0.937 | 0.953 | 1.000 |
| extrinsics 3 deg | 0.700 | 0.700 | 0.565 | 0.725 | 0.772 | 0.782 | 0.842 | 1.000 |
| mean over the conditions | 0.651 | 0.741 | 0.566 | 0.763 | 0.835 | 0.887 | 0.911 | 1.000 |

**Failure phase: point F1 @1 m in front of the vehicle only (y > 0)**

| condition after the failure | perceived graph (E68) | last good graph carried forward (E68, monitor switch) | map lanes as given (no registration) | map lanes, per-frame registration | map lanes, tracked registration (every frame) | map lanes, held registration (monitor gate) | map lanes, oracle gate | map lanes, TRUE registration (ceiling: 1 by construction) |
|---|---|---|---|---|---|---|---|---|
| night | 0.318 | 0.578 | 0.543 | 0.591 | 0.698 | 0.804 | 0.828 | 1.000 |
| 3 forward cameras off | 0.507 | 0.499 | 0.543 | 0.749 | 0.832 | 0.848 | 0.881 | 1.000 |
| blur | 0.462 | 0.602 | 0.549 | 0.663 | 0.756 | 0.862 | 0.886 | 1.000 |
| front camera off | 0.697 | 0.697 | 0.543 | 0.821 | 0.867 | 0.870 | 0.880 | 1.000 |
| fog | 0.848 | 0.842 | 0.543 | 0.889 | 0.954 | 0.947 | 0.955 | 1.000 |
| radar nearly off | 0.863 | 0.857 | 0.549 | 0.881 | 0.955 | 0.949 | 0.965 | 1.000 |
| extrinsics 3 deg | 0.710 | 0.704 | 0.549 | 0.723 | 0.759 | 0.771 | 0.828 | 1.000 |
| mean over the conditions | 0.629 | 0.683 | 0.545 | 0.759 | 0.832 | 0.865 | 0.889 | 1.000 |

**By the time since the failure began (all seven conditions pooled counts): perceived graph, carried-forward graph (E68) and map lanes under three registrations**

| seconds since the failure began | perceived graph: whole / in front | carried forward (E68): whole / in front | map lanes, held registration: whole / in front | map lanes, per-frame registration: whole / in front | map lanes as given: whole / in front |
|---|---|---|---|---|---|
| 0-1 s | 0.665 / 0.634 | 0.778 / 0.743 | 0.893 / 0.867 | 0.780 / 0.768 | 0.599 / 0.568 |
| 1-2 s | 0.670 / 0.640 | 0.769 / 0.718 | 0.890 / 0.872 | 0.797 / 0.797 | 0.623 / 0.593 |
| 2-3 s | 0.688 / 0.649 | 0.763 / 0.698 | 0.914 / 0.905 | 0.788 / 0.787 | 0.552 / 0.519 |
| 3-4 s | 0.685 / 0.648 | 0.756 / 0.689 | 0.913 / 0.890 | 0.790 / 0.769 | 0.573 / 0.531 |
| 4-5 s | 0.683 / 0.651 | 0.745 / 0.676 | 0.908 / 0.882 | 0.787 / 0.773 | 0.568 / 0.541 |
| 5-6 s | 0.670 / 0.634 | 0.728 / 0.651 | 0.894 / 0.847 | 0.758 / 0.740 | 0.557 / 0.564 |
| 6-7 s | 0.673 / 0.639 | 0.722 / 0.645 | 0.884 / 0.845 | 0.758 / 0.749 | 0.530 / 0.509 |
| 7-8 s | 0.676 / 0.640 | 0.714 / 0.632 | 0.894 / 0.857 | 0.760 / 0.763 | 0.544 / 0.519 |
| 8-9 s | 0.677 / 0.648 | 0.725 / 0.660 | 0.856 / 0.832 | 0.713 / 0.715 | 0.520 / 0.514 |
| 9-10 s | 0.665 / 0.643 | 0.710 / 0.658 | 0.868 / 0.857 | 0.737 / 0.734 | 0.573 / 0.563 |

**Outdated map and tolerance (held registration; failure phase, all conditions pooled): what an outdated map and a looser tolerance do**

| map used as the lane graph | held registration, @1 m tolerance: whole / in front | held registration, @1.5 m: whole / in front | true registration, @1 m: whole / in front (what staleness alone costs) |
|---|---|---|---|
| current map | 0.891 / 0.866 | 0.960 / 0.948 | 1.000 / 1.000 |
| 30 % of the lanes removed (outdated) | 0.773 / 0.750 | 0.844 / 0.833 | 0.870 / 0.870 |
| 30 % of the lanes shifted by 1.5 m (re-painted) | 0.725 / 0.721 | 0.871 / 0.870 | 0.757 / 0.771 |

**Detecting an outdated map: the carried-forward graph (source frame at most 3 s old; 1908 failure-phase frames of the seven conditions) against the map lanes placed by the held registration**

| outdated map variant (against the current map, same frames) | signal: share of the carried-forward graph within 1 m of a map lane - median, current map | median, outdated map | 10 % quantile current / 90 % quantile outdated | AUROC 'outdated vs current', single frame | AUROC, mean of 5 consecutive frames (2.5 s) |
|---|---|---|---|---|---|
| 30 % of the lanes removed | 0.896 | 0.657 | 0.728 / 0.811 | 0.901 | 0.958 (363 windows) |
| 30 % of the lanes shifted by 1.5 m | 0.896 | 0.715 | 0.728 / 0.848 | 0.860 | 0.928 (363 windows) |

**Choosing the source of the lane geometry during a failure: fixed sources and two selection rules (point F1 @1 m, failure phase, all seven conditions pooled; 'wrong map' = the map of another place, shown in this frame)**

| source of the lane geometry (failure phase, all seven conditions pooled) | current map: F1 whole / in front | 30 % of the lanes removed: F1 whole / in front | 30 % of the lanes shifted by 1.5 m: F1 whole / in front | WRONG map (the map of another place): F1 whole / in front |
|---|---|---|---|---|
| perceived graph only | 0.675 / 0.642 | 0.675 / 0.642 | 0.675 / 0.642 | 0.675 / 0.642 |
| last good graph carried forward when the monitor flags (E68) | 0.744 / 0.681 | 0.744 / 0.681 | 0.744 / 0.681 | 0.744 / 0.681 |
| map lanes (held registration) whenever the monitor flags | 0.808 / 0.761 | 0.772 / 0.728 | 0.759 / 0.724 | 0.600 / 0.571 |
| rule A: map lanes if the carried-forward graph agrees with them (no signal for sources older than 3 s: trust the map) | 0.805 / 0.758 | 0.766 / 0.720 | 0.758 / 0.719 | 0.664 / 0.625 |
| rule B: map lanes if the map-validity signal is high, else the carried-forward graph | 0.793 / 0.748 | 0.744 / 0.681 | 0.744 / 0.681 | 0.744 / 0.681 |
| oracle: best of the three sources per frame | 0.907 / 0.884 | 0.826 / 0.797 | 0.803 / 0.775 | 0.747 / 0.691 |
| share of the flagged frames in which rule A / rule B chose the map lanes | 0.92 / 0.84 | 0.72 / 0.00 | 0.81 / 0.00 | 0.55 / 0.00 |

**The map-validity signal: agreement (share of the points within 1 m of a map lane) between the PERCEIVED graph of the last 10 healthy frames (monitor p <= 0.5) and the map lanes placed by the registration; rule B's threshold is the 5 % quantile of this signal on the clean frames of the current map (0.869; 2273 clean frames, no labels)**

| map used | median map validity, current map (same frames) | median map validity, this map | AUROC 'not the current map' from the validity | frames |
|---|---|---|---|---|
| 30 % of the lanes removed | 0.918 | 0.699 | 0.998 | 738 |
| 30 % of the lanes shifted by 1.5 m | 0.918 | 0.709 | 0.992 | 738 |
| WRONG map (the map of another place) | 0.918 | 0.269 | 1.000 | 738 |

## Result images

![Point F1 @1 m of the lane graph in the failure phase: perceived graph, carried-forward graph (E68) and the map lanes placed by the registration of E67 (whole BEV left, in front of the vehicle right). The 'true registration' bar is 1 by construction (the ground truth is the map).](e69_bars.png)
*Point F1 @1 m of the lane graph in the failure phase: perceived graph, carried-forward graph (E68) and the map lanes placed by the registration of E67 (whole BEV left, in front of the vehicle right). The 'true registration' bar is 1 by construction (the ground truth is the map).*

![Point F1 @1 m against the seconds since the failure began: the map lanes under the held registration are better than the carried-forward graph at every moment (0.89 against 0.78 in the first second) and age only slowly (their level is set by the registration offset of about 1 m against the 1 m tolerance), while the carried-forward graph decays, faster in front of the vehicle.](e69_time.png)
*Point F1 @1 m against the seconds since the failure began: the map lanes under the held registration are better than the carried-forward graph at every moment (0.89 against 0.78 in the first second) and age only slowly (their level is set by the registration offset of about 1 m against the 1 m tolerance), while the carried-forward graph decays, faster in front of the vehicle.*

![Night, a moving vehicle, 1.5 s / 5 s / 9 s after the failure began: ground truth (green), the map lanes as given (grey dotted, biased), the same lanes placed by the held registration (blue) and the graph carried forward from the last good frame (orange).](e69_bev.png)
*Night, a moving vehicle, 1.5 s / 5 s / 9 s after the failure began: ground truth (green), the map lanes as given (grey dotted, biased), the same lanes placed by the held registration (blue) and the graph carried forward from the last good frame (orange).*

## Findings
- With a current map, the map lanes placed by the held registration are a far better lane graph than the failing perception and than the carried-forward graph. Mean over the seven failure types, failure phase, point F1 @1 m for the whole BEV (in front of the vehicle in brackets): perceived graph 0.651 (0.629), graph carried forward (E68) 0.741 (0.683), map lanes with the held registration 0.887 (0.865), with an oracle gate 0.911 (0.889); at night 0.293 (0.318) -> 0.688 (0.578) -> 0.843 (0.804), under blur 0.474 (0.462) -> 0.697 (0.602) -> 0.895 (0.862). The ceiling is 1.0 by construction (the ground truth is the map).
- The registration is what makes the map lanes usable: as given they reach only 0.566 (0.545) - the biased map is on average 2.6 m off at the vehicle (E67), so most lane points miss the 1 m tolerance; the per-frame registration gives 0.763 (0.759) - at night 0.579, because it follows the failing graph; the tracked registration 0.835 (0.832); the held one 0.887 (0.865). This is the ranking of E67 (lateral error 1.32 / 0.60 / 0.45 / 0.39 m) expressed in lane-graph F1; the oracle gate adds 0.024.
- Over time the held map lanes age slowly while the carried-forward graph decays: pooled over the conditions the map lanes are at 0.893 in the first second after the failure began and 0.868 after nine seconds (night 0.90 -> 0.75 over ten seconds, blur 0.91 -> 0.84; in front of the vehicle 0.867 -> 0.857 pooled), the carried-forward graph at 0.778 -> 0.710 (in front 0.743 -> 0.658). The map lanes are better at every moment of the failure, including the first second (0.893 against 0.778), because the registration was tracked from clean frames before the failure and the map covers the whole +-48 m around the vehicle, including the area ahead that the old graph never saw. The carried-forward graph is therefore the fallback for the case that there is no reliable map - and an independent cross-check - not the first layer.
- An outdated map costs what it costs. Pooled over the failure frames, with 30 % of the lanes removed the held-registration F1 is 0.773 (0.750 in front), with 30 % of the lanes re-painted (shifted by 1.5 m) 0.725 (0.721) - the level of the carried-forward graph in the whole BEV and above it in front of the vehicle - against 0.891 (0.866) for the current map; with the TRUE registration the staleness alone gives 0.870 (removed) and 0.757 (shifted). The map lanes are worth using as long as roughly 70 % of the lanes are right. With a 1.5 m instead of a 1 m tolerance the held-registration F1 is 0.960 (current), 0.844 (removed) and 0.871 (shifted): the 1 m tolerance is demanding for a registration whose offset is about 1 m (E67), and at the looser tolerance the staleness rather than the registration limits the result.
- The disagreement between the carried-forward graph and the map lanes detects an outdated map. Signal: the share of the carried-forward graph (source frame at most 3 s old) that lies within 1 m of a map lane placed by the held registration. On the same 1,908 failure-phase frames the median is 0.896 with the current map, 0.657 with 30 % of the lanes removed and 0.715 with 30 % shifted by 1.5 m; the AUROC for 'outdated against current' is 0.901 / 0.860 per frame and 0.958 / 0.928 when five consecutive frames (2.5 s) are averaged. It needs no ground truth and no extra network, so the system can choose its lane source: the map lanes while the agreement is high, the carried-forward graph (for a few seconds) and a warning to the planner when it is not. Caveats: the staleness is simulated (random lanes), the carried-forward graph has errors of its own (F1 0.74), and in a long outage the old graph no longer covers the area ahead.
- Choosing the source with a map-validity signal. A better, label-free signal is available BEFORE the failure: the agreement between the perceived graph of the last ten healthy frames (monitor p <= 0.5) and the map lanes placed by the registration, i.e. the share of the perceived points within 1 m of a map lane. Its median is 0.918 with the current map, 0.699 with 30 % of the lanes removed, 0.709 with 30 % re-painted and 0.269 with the map of another place; the AUROC for 'not the current map' is 0.998 / 0.992 / 1.000 (738 frames). Used as a rule (rule B: take the map lanes while the validity is above the 5 % quantile of the clean frames, 0.869, otherwise the carried-forward graph) the pooled F1 in the failure phase is 0.793 (in front 0.748) with the current map - against 0.808 for taking the map lanes whenever the monitor flags and 0.744 for the carried-forward graph -, 0.744 with an outdated map (it rejects the stale maps, which would have given 0.772 / 0.759) and 0.744 with a WRONG map, where taking the map lanes whenever the monitor flags gives 0.600. The rule costs 0.015 F1 where the map is current, up to 0.03 where it is outdated, and avoids a loss of 0.14 where it is wrong. Rule A, which compares the carried-forward graph with the map lanes, is weaker (0.805 / 0.766 / 0.758 / 0.664): it has a signal only while the source frame is at most 3 s old and trusts the map otherwise, so it cannot protect against a wrong map in a long outage. The oracle choice of the best source per frame reaches 0.907 / 0.826 / 0.803 / 0.747.
- Reading for the system: during a perception outage the lane geometry has two sources with different failure modes - the map lanes placed by the held registration (best with a current map, aging with the registration, wrong where the map is outdated) and the carried-forward graph (needs no map, decays within about 5 s in front of the vehicle). Together with E67 the planner gets during an outage a route reference (0.39 m) and lane geometry from the map and the held registration (F1 0.89 with a current map, 0.72-0.77 with an outdated one), with the carried-forward graph as the independent, map-free cross-check; the agreement of the map lanes with what was seen before is a usable warning that the map is outdated or wrong (the two findings above).

## Limits
- Circular by construction: on nuScenes the ground-truth lane graph is the map, so the map lanes with the true registration reach 1.0 and every number here measures what the registration error and the (simulated) staleness of the map cost, not the benefit of an independent map. In a real vehicle the map is an independent description of the road that may differ from reality (construction, repainting); the two outdated-map variants emulate this crudely (random removal / shifting of single lanes, whereas real changes are structured and local), and the registrations were computed with the current map, not with the outdated one.
- The registration error comes from the E61 stress model (a rigid global map error with a yaw about the scene start, 1 m / 2 deg, growing with the distance driven; the held registration then ages with the distance) - harsher than a GNSS/INS-aided localisation but without jumps; the failure switches on abruptly in the middle of every scene and lasts up to 10 s (E67). The map lanes carry no dynamic information (closed lanes, obstacles) and no topology is evaluated (the metric is the point F1 @1 m).
- The rule thresholds are 5 % quantiles of the signals on the clean TEST frames of the current map (no labels, but TEST data; a deployed system would calibrate them on a drive); the map-validity signal needs healthy frames with a trusted registration before the failure (a drive that starts with failed sensors has none, and the rule then defaults to the map lanes). The wrong map is the map of a random sample of another scene shown in this frame; real wrong-map situations (a wrong tile after a localisation failure) would look similar.
- The comparison with the carried-forward graph uses the E68 numbers of the same frames; the tolerance of 1 m is strict for a registration offset of about 1 m, the 1.5 m numbers show the sensitivity. Single scenario per condition on 21 TEST scenes, frames of a scene strongly correlated.
