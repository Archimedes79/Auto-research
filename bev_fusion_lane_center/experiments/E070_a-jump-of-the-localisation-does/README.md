# E70: A jump of the localisation: does the tracked registration follow it, and does a score-triggered restart help?

Status: completed

**Question:** E61 and E67 model the localisation error of the map as a scene-constant bias with jitter (and, in a variant, a slow drift). Real localisation also JUMPS - a GNSS fix after an outage, a wrong lane match, a map tile change. The tracked registration searches only in a small window (+-1 m) around its prediction, so a jump larger than the window could leave it stuck on a wrong local optimum. Does it follow a sudden jump of the map offset (1.5 - 10 m, with or without a yaw jump), how long does it take, and does a label-free restart rule - run a wide search when the registration score of the window falls below the 5 % quantile of the scores before the jump - recover at once, at what cost and with how many false alarms?

## Setup
- Training / test diagnosis: nothing is trained. The only choices are the window of the tracker (+-1 m, +-3 deg; E61, K = 5 frames), the wide search of the restart (+-8 m, +-8 deg, step 2 m / 4 deg, then ICP) and the restart threshold, which is the 5 % quantile of the window scores on the frames BEFORE the jump (0.632; no labels). TEST only; clean sensors (the clean graphs of E57) so that the registration can be judged on its own.
- Scenario: the 21 TEST scenes, frames sorted by time; the map has the E61 pose error (scene-constant offset N(0, 1 m) per axis and yaw N(0, 2 deg) about the scene start, jitter 0.2 m / 0.3 deg); in the MIDDLE of every scene the global offset jumps by J metres in a random direction (J = 1.5 / 3 / 6 / 10 m, and a variant of 3 m with a 4 deg yaw jump). The frames before the jump are the seen condition, the frames after it the unseen one.
- Methods: the map route as given; the per-frame graph-ICP (full search +-3 m, +-6 deg; E59); the tracked registration of E61 / E67 (window search around the prediction from the median of the last 5 estimates) WITHOUT a restart; the same tracker WITH the restart (if the window score is below the threshold, a wide search is run and, if it scores better, replaces the estimate and the history is restarted); the true registration (the floor). Metrics: anchored lateral error of the registered map route (E62) and the offset error of the registration at the vehicle, by the time since the jump (0.5 s per frame); the number of frames until the offset error is below 1 m (and the share of scenes that never get there); the restart events and the false alarms before the jump.

![architecture](E70_architecture.png)

## Numbers

**Lateral error (m, anchored, mean over 1 / 2 / 3 s) of the registered map route by the time since the jump (0.5 s per frame; 21 TEST scenes, clean sensors); a jump of 10 m is beyond the +-8 m search of the restart**

| jump | method | lateral error (m): before the jump | lateral error (m): 0-1.5 s after | lateral error (m): 1.5-3 s | lateral error (m): 3-5 s | lateral error (m): 5 s or more |
|---|---|---|---|---|---|---|
| 1.5 m | map as given | 0.85 | 1.41 | 1.53 | 1.61 | 1.83 |
| 1.5 m | per-frame (+-3 m) | 0.26 | 0.41 | 0.40 | 0.42 | 0.53 |
| 1.5 m | tracked, no restart | 0.23 | 0.28 | 0.27 | 0.34 | 0.29 |
| 1.5 m | tracked + restart | 0.23 | 0.26 | 0.27 | 0.34 | 0.29 |
| 1.5 m | true (floor) | 0.23 | 0.21 | 0.25 | 0.30 | 0.28 |
| 3 m | map as given | 0.85 | 1.76 | 1.85 | 1.97 | 2.26 |
| 3 m | per-frame (+-3 m) | 0.26 | 0.56 | 0.61 | 0.62 | 0.70 |
| 3 m | tracked, no restart | 0.23 | 0.58 | 0.48 | 0.46 | 0.38 |
| 3 m | tracked + restart | 0.23 | 0.25 | 0.29 | 0.34 | 0.29 |
| 3 m | true (floor) | 0.23 | 0.21 | 0.25 | 0.30 | 0.28 |
| 6 m | map as given | 0.85 | 2.58 | 2.56 | 2.75 | 3.14 |
| 6 m | per-frame (+-3 m) | 0.26 | 1.03 | 1.00 | 0.96 | 0.81 |
| 6 m | tracked, no restart | 0.23 | 1.54 | 1.13 | 0.88 | 0.83 |
| 6 m | tracked + restart | 0.23 | 0.25 | 0.26 | 0.33 | 0.29 |
| 6 m | true (floor) | 0.23 | 0.21 | 0.25 | 0.30 | 0.28 |
| 10 m | map as given | 0.85 | 3.92 | 3.73 | 4.00 | 4.46 |
| 10 m | per-frame (+-3 m) | 0.26 | 1.89 | 1.84 | 1.76 | 1.76 |
| 10 m | tracked, no restart | 0.23 | 2.84 | 1.98 | 1.31 | 0.88 |
| 10 m | tracked + restart | 0.23 | 0.55 | 0.38 | 0.47 | 0.41 |
| 10 m | true (floor) | 0.23 | 0.21 | 0.25 | 0.30 | 0.28 |
| 3 m + 4 deg | map as given | 0.85 | 2.81 | 3.19 | 3.58 | 4.28 |
| 3 m + 4 deg | per-frame (+-3 m) | 0.26 | 0.80 | 0.85 | 1.03 | 1.79 |
| 3 m + 4 deg | tracked, no restart | 0.23 | 1.02 | 0.59 | 0.52 | 0.49 |
| 3 m + 4 deg | tracked + restart | 0.23 | 0.30 | 0.27 | 0.33 | 0.30 |
| 3 m + 4 deg | true (floor) | 0.23 | 0.21 | 0.25 | 0.30 | 0.28 |

**Registration offset error (m) at the vehicle, by the time since the jump**

| jump | method | registration offset error (m): before the jump | registration offset error (m): 0-1.5 s after | registration offset error (m): 1.5-3 s | registration offset error (m): 3-5 s | registration offset error (m): 5 s or more |
|---|---|---|---|---|---|---|
| 1.5 m | map as given | 1.49 | 2.62 | 2.76 | 3.02 | 3.54 |
| 1.5 m | per-frame (+-3 m) | 0.77 | 1.02 | 0.95 | 1.08 | 1.54 |
| 1.5 m | tracked, no restart | 0.72 | 0.72 | 0.54 | 0.57 | 0.66 |
| 1.5 m | tracked + restart | 0.69 | 0.95 | 0.72 | 0.68 | 0.77 |
| 1.5 m | true (floor) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 3 m | map as given | 1.49 | 3.72 | 3.84 | 4.01 | 4.45 |
| 3 m | per-frame (+-3 m) | 0.77 | 1.18 | 1.15 | 1.27 | 1.56 |
| 3 m | tracked, no restart | 0.72 | 1.64 | 1.30 | 1.02 | 1.09 |
| 3 m | tracked + restart | 0.69 | 1.11 | 0.88 | 0.71 | 0.76 |
| 3 m | true (floor) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 6 m | map as given | 1.49 | 6.25 | 6.34 | 6.46 | 6.78 |
| 6 m | per-frame (+-3 m) | 0.77 | 2.42 | 2.58 | 2.63 | 2.73 |
| 6 m | tracked, no restart | 0.72 | 4.46 | 3.26 | 2.05 | 1.91 |
| 6 m | tracked + restart | 0.69 | 1.18 | 0.82 | 0.60 | 0.69 |
| 6 m | true (floor) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 10 m | map as given | 1.49 | 10.07 | 10.09 | 10.13 | 10.38 |
| 10 m | per-frame (+-3 m) | 0.77 | 5.59 | 5.82 | 5.84 | 6.30 |
| 10 m | tracked, no restart | 0.72 | 8.24 | 6.40 | 4.75 | 2.86 |
| 10 m | tracked + restart | 0.69 | 1.67 | 1.08 | 0.98 | 1.01 |
| 10 m | true (floor) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 3 m + 4 deg | map as given | 1.49 | 4.57 | 5.04 | 5.82 | 7.39 |
| 3 m + 4 deg | per-frame (+-3 m) | 0.77 | 1.80 | 1.72 | 2.10 | 3.49 |
| 3 m + 4 deg | tracked, no restart | 0.72 | 2.36 | 1.68 | 1.18 | 1.15 |
| 3 m + 4 deg | tracked + restart | 0.69 | 0.91 | 0.67 | 0.60 | 0.79 |
| 3 m + 4 deg | true (floor) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

**Recovery: frames until the offset error is below 1 m, share of scenes that never recover, and restart events**

| jump | tracked: median frames to recover (offset < 1 m) / never | tracked + restart: median frames / never | per-frame: median frames / never | restart events (before the jump = false alarms) |
|---|---|---|---|---|
| 1.5 m | 0 / 0.00 | 0 / 0.00 | 0 / 0.10 | 15 (3) |
| 3 m | 2 / 0.05 | 0 / 0.05 | 0 / 0.14 | 21 (3) |
| 6 m | 7 / 0.24 | 0 / 0.00 | 13 / 0.43 | 34 (3) |
| 10 m | 18 / 0.43 | 1 / 0.05 | 99 / 0.81 | 33 (3) |
| 3 m + 4 deg | 2 / 0.14 | 0 / 0.00 | 0 / 0.29 | 26 (3) |

## Result images

![Lateral error of the registered map route before and after a jump of the map offset (clean sensors; the jump happens in the middle of every TEST scene).](E70_e70_lines.png)
*Lateral error of the registered map route before and after a jump of the map offset (clean sensors; the jump happens in the middle of every TEST scene).*

![Registration offset error per frame in three scenes around a 3 m jump: the tracked registration without a restart converges slowly, the score-triggered restart within a frame or two; the per-frame registration is noisy throughout.](E70_e70_trace.png)
*Registration offset error per frame in three scenes around a 3 m jump: the tracked registration without a restart converges slowly, the score-triggered restart within a frame or two; the per-frame registration is noisy throughout.*

## Findings
- A small jump is followed, a large one is not - without a restart. The tracked registration follows a 1.5 m jump without loss (lateral error 0.28 m in the first 1.5 s against 0.21-0.30 m for the true registration), needs three to five seconds to settle after a 3 m jump (lateral error 0.58 m in the first 1.5 s, 0.46 m at 3-5 s, 0.38 m beyond; the median offset is below 1 m after 1 s, and 5 % of the scenes never get there) and longer for 6 m (1.54 / 1.13 / 0.88 / 0.83 m; median 3.5 s, 24 % of the scenes never) and leaves 43 % of the scenes unrecovered after a 10 m jump (2.84 -> 0.88 m). The mechanism: the +-1 m window around a wrong prediction does not contain the optimum, the registration score drops, and the ICP stage pulls the estimate back only slowly through the history of five frames.
- The score-triggered restart recovers at once. In the first 1.5 s after the jump the lateral error is 0.25 m for a 3 m jump, 0.25 m for 6 m and 0.55 m for 10 m (beyond the +-8 m search, where the ICP stage still gets most of the way) against 0.58 / 1.54 / 2.84 m without the restart - at the level of the true registration (0.21 m) from the first or second frame: the median frames to an offset below 1 m is 0, the share of scenes that never recover 0 % (6 m) and 5 % (3 m and 10 m). With a 3 m offset jump together with a 4 deg yaw jump the lateral error is 1.02 / 0.59 / 0.52 / 0.49 m without and 0.30 / 0.27 / 0.33 / 0.30 m with the restart.
- The rule is label-free and cheap. The threshold is the 5 % quantile of the window scores before the jump (0.632); it produced 3 restarts before the jump in the 21 scenes (about 0.7 % of the frames; each costs one wide search of roughly 70 ms of CPU instead of 28 ms for the window), and 12-31 restarts after the jump (0.6 to 1.5 per scene, mostly in the first frames after the jump). The registration score that drives it is the same quantity that gates the hold in E67 - there a low score caused by a failing PERCEPTION is answered by holding the registration, here a low score while the perception is healthy is answered by a restart; the monitor of E64 tells the two cases apart (the combination is tested in E71).
- The per-frame registration with its +-3 m full search cannot follow jumps beyond its search range: 14 % of the scenes never recover from a 3 m jump, 43 % from 6 m and 81 % from 10 m, and it is noisier than the tracker even where it works (0.56-0.70 m after a 3 m jump against 0.38-0.58 m). The tracked registration with the restart is better than either mechanism alone: the smoothing of the tracker when nothing happens, the search of the restart when something does.
- Reading for the system: the localisation error of the map is not only a slowly varying quantity (E61, E67) but may jump (GNSS fix after an outage, wrong lane match, tile change). A tracker needs a restart rule; the registration score is a sufficient, label-free trigger, a wide search of +-8 m costs about as much as the full search of E59, and false alarms are rare. What no perception-based rule can do is detect or correct a jump that happens while the sensors are blind (E67 / E68): then only odometry or GNSS consistency can tell, and the held registration is wrong.

## Limits
- Clean sensors throughout (so that the registration is judged on its own): a jump during a sensor failure cannot be handled from the perception and is not covered here (E71 tests it together with a failure). The jump is instantaneous, in the middle of every scene, with a random direction and one magnitude per run; real jumps have structure (along the road after a GNSS fix) and may be followed by drift.
- The pose model is the E61 stress model; the restart threshold is the 5 % quantile of the window scores on the TEST frames before the jump (no labels, but TEST data); the wide search covers +-8 m and +-8 deg (a jump of 10 m is mostly recovered by the ICP stage, larger ones would need a global re-localisation against the map). Frames are 0.5 s apart; 21 TEST scenes; the lateral error is the open-loop proxy of E62.
- The CPU times are those of the unoptimised Python implementation (median of 30 frames: 28 ms for the window search, 69 ms for the full +-3 m search and 72 ms for the wide +-8 m search; 95th percentile 42 / 109 / 122 ms).
