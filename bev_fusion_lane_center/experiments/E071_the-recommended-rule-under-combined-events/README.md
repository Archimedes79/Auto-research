# E71: The recommended rule under combined events: a sensor failure and a jump of the localisation

Status: completed

**Question:** E67 holds the registration while the monitor reports a failure, E70 restarts it when the registration score drops although the perception is healthy. Together they form the recommended rule for the map registration: HOLD while the monitor fires, RESTART when the monitor says healthy and the window score is low, otherwise TRACK - plus a check at the return of the sensors. Does the combination work when both events happen in one scene - in particular when the localisation jumps WHILE the sensors are blind and the registration has to be recovered when they return - and does a restart without the monitor gate really hurt during a failure?

## Setup
- Training / test diagnosis: nothing is trained. The monitor is the E64 regression (cross-fitted over the frame parity, as in E67); the restart threshold is the 5 % quantile of the window scores on clean frames without events (0.631; label-free; E70); the wide search is +-8 m / +-8 deg. TEST only (21 scenes; the frames before the first event are the seen condition, the failure the unseen one).
- Scenario: in each TEST scene the sensors fail from 30 % to 70 % of the scene (the perturbed graphs of E57 - night, three forward cameras off, blur, front camera off; the frames after 70 % are clean again) and the map offset jumps by 3 m (6 m in one run) in a random direction at 15 % of the scene (BEFORE the failure, healthy sensors), at 50 % (DURING the failure: the sensors are blind) or not at all (the transient failure of E67). The map has the E61 pose error (scene-constant offset and yaw bias plus jitter). The clean frames before and after the failure are the same for all four failure types, so those columns are not independent across failure types.
- Methods (registration of the map route): the per-frame graph-ICP (full search +-3 m); the tracked registration of E61 (window +-1 m around the prediction) updated at every frame; the tracked registration with the monitor gate and HOLD (E67); 'hold + restart' (a restart is allowed only on frames the monitor rates healthy, p <= 0.5, and only if the window score is below the threshold); the RECOMMENDED rule 'hold + restart + verify' (additionally, when the monitor says healthy again after at least four held frames = 2 s, the held registration is checked with the wide search and replaced if the wide search scores at least as well, within 0.05); a 'strict verify' variant of it that replaces the held registration only if the wide search scores better than the held one; a restart on a low score WITHOUT the monitor gate; the true registration (floor). Metrics: anchored lateral error of the registered map route (E62) and the registration offset error, by phase (before the failure / during / after, by the time since the sensors returned).

![architecture](E71_architecture.png)

## Numbers

**Lateral error (m, anchored) of the registered map route by phase (mean over four failure types, the 1.5-3 s column over three because the blur scenes have no frames there; 21 TEST scenes). Event: no jump / a 3 m jump BEFORE the failure (at 15 % of the scene) / a 3 m or 6 m jump DURING the failure (at 50 %, the sensors are blind)**

| event | method | lateral error (m): before the failure | lateral error (m): during the failure | lateral error (m): after: 0-1.5 s | lateral error (m): after: 1.5-3 s |
|---|---|---|---|---|---|
| no jump | per-frame registration | 0.22 | 0.62 | 0.52 | 0.47 |
| no jump | tracked (E61) | 0.21 | 0.50 | 0.35 | 0.33 |
| no jump | tracked + hold (E67) | 0.20 | 0.38 | 0.36 | 0.34 |
| no jump | hold + restart (E70) | 0.20 | 0.38 | 0.36 | 0.34 |
| no jump | hold + restart + verify (recommended) | 0.20 | 0.38 | 0.34 | 0.33 |
| no jump | hold + restart + strict verify | 0.20 | 0.38 | 0.35 | 0.34 |
| no jump | restart without monitor gate | 0.21 | 0.74 | 0.35 | 0.32 |
| no jump | true registration (floor) | 0.21 | 0.25 | 0.31 | 0.28 |
| 3 m before | per-frame registration | 0.22 | 0.79 | 0.69 | 0.66 |
| 3 m before | tracked (E61) | 0.29 | 0.53 | 0.39 | 0.33 |
| 3 m before | tracked + hold (E67) | 0.34 | 0.61 | 0.52 | 0.42 |
| 3 m before | hold + restart (E70) | 0.25 | 0.49 | 0.42 | 0.39 |
| 3 m before | hold + restart + verify (recommended) | 0.25 | 0.50 | 0.40 | 0.38 |
| 3 m before | hold + restart + strict verify | 0.25 | 0.49 | 0.40 | 0.38 |
| 3 m before | restart without monitor gate | 0.20 | 0.69 | 0.36 | 0.32 |
| 3 m before | true registration (floor) | 0.21 | 0.25 | 0.31 | 0.28 |
| 3 m during | per-frame registration | 0.22 | 0.72 | 0.69 | 0.66 |
| 3 m during | tracked (E61) | 0.21 | 0.62 | 0.54 | 0.40 |
| 3 m during | tracked + hold (E67) | 0.20 | 0.61 | 0.66 | 0.51 |
| 3 m during | hold + restart (E70) | 0.20 | 0.57 | 0.51 | 0.41 |
| 3 m during | hold + restart + verify (recommended) | 0.20 | 0.56 | 0.43 | 0.38 |
| 3 m during | hold + restart + strict verify | 0.20 | 0.56 | 0.43 | 0.38 |
| 3 m during | restart without monitor gate | 0.21 | 0.73 | 0.37 | 0.32 |
| 3 m during | true registration (floor) | 0.21 | 0.25 | 0.31 | 0.28 |
| 6 m during | per-frame registration | 0.22 | 0.94 | 0.95 | 0.87 |
| 6 m during | tracked (E61) | 0.21 | 0.89 | 0.89 | 0.64 |
| 6 m during | tracked + hold (E67) | 0.20 | 0.99 | 1.25 | 0.94 |
| 6 m during | hold + restart (E70) | 0.20 | 0.79 | 0.56 | 0.44 |
| 6 m during | hold + restart + verify (recommended) | 0.20 | 0.79 | 0.56 | 0.44 |
| 6 m during | hold + restart + strict verify | 0.20 | 0.79 | 0.55 | 0.44 |
| 6 m during | restart without monitor gate | 0.21 | 0.79 | 0.37 | 0.32 |
| 6 m during | true registration (floor) | 0.21 | 0.25 | 0.31 | 0.28 |

**Registration offset error (m) at the vehicle by phase (mean over four failure types; events as above)**

| event | method | offset error (m): before | offset error (m): during | offset error (m): after 0-1.5 s | offset error (m): after 1.5-3 s | offset error (m): after 3-5 s | offset error (m): after 5 s or more |
|---|---|---|---|---|---|---|---|
| no jump | per-frame registration | 0.72 | 1.78 | 1.21 | 1.35 | 1.57 | 1.63 |
| no jump | tracked (E61) | 0.73 | 1.10 | 0.65 | 0.75 | 0.79 | 0.71 |
| no jump | tracked + hold (E67) | 0.71 | 0.80 | 0.60 | 0.73 | 0.76 | 0.64 |
| no jump | hold + restart (E70) | 0.71 | 0.81 | 0.61 | 0.73 | 0.76 | 0.63 |
| no jump | hold + restart + verify (recommended) | 0.71 | 0.89 | 0.68 | 0.77 | 0.73 | 0.64 |
| no jump | hold + restart + strict verify | 0.71 | 0.85 | 0.67 | 0.77 | 0.75 | 0.63 |
| no jump | restart without monitor gate | 0.72 | 2.23 | 0.89 | 0.92 | 0.83 | 0.73 |
| no jump | true registration (floor) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 3 m before | per-frame registration | 0.76 | 2.00 | 1.46 | 1.49 | 1.50 | 1.54 |
| 3 m before | tracked (E61) | 1.02 | 1.41 | 0.97 | 0.94 | 0.95 | 0.76 |
| 3 m before | tracked + hold (E67) | 1.13 | 1.47 | 1.08 | 1.07 | 0.99 | 0.78 |
| 3 m before | hold + restart (E70) | 0.89 | 1.17 | 0.75 | 0.80 | 0.80 | 0.72 |
| 3 m before | hold + restart + verify (recommended) | 0.89 | 1.25 | 0.89 | 0.87 | 0.74 | 0.59 |
| 3 m before | hold + restart + strict verify | 0.89 | 1.19 | 0.82 | 0.85 | 0.79 | 0.72 |
| 3 m before | restart without monitor gate | 0.80 | 2.22 | 0.87 | 0.81 | 0.79 | 0.70 |
| 3 m before | true registration (floor) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 3 m during | per-frame registration | 0.72 | 1.91 | 1.46 | 1.49 | 1.50 | 1.54 |
| 3 m during | tracked (E61) | 0.73 | 1.54 | 1.24 | 1.16 | 1.07 | 0.93 |
| 3 m during | tracked + hold (E67) | 0.71 | 1.59 | 1.50 | 1.38 | 1.23 | 1.12 |
| 3 m during | hold + restart (E70) | 0.71 | 1.54 | 1.22 | 1.04 | 0.88 | 0.81 |
| 3 m during | hold + restart + verify (recommended) | 0.71 | 1.54 | 0.99 | 0.89 | 0.80 | 0.64 |
| 3 m during | hold + restart + strict verify | 0.71 | 1.53 | 1.01 | 0.89 | 0.83 | 0.77 |
| 3 m during | restart without monitor gate | 0.72 | 2.42 | 0.92 | 0.83 | 0.83 | 0.75 |
| 3 m during | true registration (floor) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 6 m during | per-frame registration | 0.72 | 2.63 | 2.75 | 2.66 | 2.78 | 2.90 |
| 6 m during | tracked (E61) | 0.73 | 2.56 | 2.39 | 1.79 | 1.60 | 1.48 |
| 6 m during | tracked + hold (E67) | 0.71 | 2.82 | 3.26 | 2.59 | 2.05 | 1.82 |
| 6 m during | hold + restart (E70) | 0.71 | 2.31 | 1.55 | 1.31 | 0.88 | 0.70 |
| 6 m during | hold + restart + verify (recommended) | 0.71 | 2.28 | 1.28 | 1.17 | 0.84 | 0.67 |
| 6 m during | hold + restart + strict verify | 0.71 | 2.29 | 1.22 | 1.11 | 0.82 | 0.67 |
| 6 m during | restart without monitor gate | 0.72 | 2.60 | 0.83 | 0.86 | 0.86 | 0.76 |
| 6 m during | true registration (floor) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

**Per failure type (3 m jump during the failure)**

| condition | method | lateral error (m): before the failure | lateral error (m): during the failure | lateral error (m): after: 0-1.5 s | lateral error (m): after: 1.5-3 s |
|---|---|---|---|---|---|
| night | per-frame registration | 0.22 | 0.98 | 0.69 | 0.66 |
| night | tracked (E61) | 0.21 | 0.76 | 0.72 | 0.57 |
| night | tracked + hold (E67) | 0.20 | 0.74 | 0.91 | 0.71 |
| night | hold + restart (E70) | 0.20 | 0.73 | 0.59 | 0.42 |
| night | hold + restart + verify (recommended) | 0.20 | 0.73 | 0.53 | 0.41 |
| night | hold + restart + strict verify | 0.20 | 0.73 | 0.52 | 0.41 |
| night | restart without monitor gate | 0.20 | 1.17 | 0.46 | 0.32 |
| 3 forward cameras off | per-frame registration | 0.22 | 0.53 | 0.69 | 0.66 |
| 3 forward cameras off | tracked (E61) | 0.21 | 0.52 | 0.35 | 0.31 |
| 3 forward cameras off | tracked + hold (E67) | 0.20 | 0.54 | 0.48 | 0.41 |
| 3 forward cameras off | hold + restart (E70) | 0.20 | 0.48 | 0.47 | 0.42 |
| 3 forward cameras off | hold + restart + verify (recommended) | 0.20 | 0.47 | 0.45 | 0.42 |
| 3 forward cameras off | hold + restart + strict verify | 0.20 | 0.47 | 0.45 | 0.42 |
| 3 forward cameras off | restart without monitor gate | 0.20 | 0.42 | 0.33 | 0.32 |
| blur | per-frame registration | 0.23 | 0.82 | 0.70 | - |
| blur | tracked (E61) | 0.21 | 0.75 | 0.64 | - |
| blur | tracked + hold (E67) | 0.20 | 0.68 | 0.70 | - |
| blur | hold + restart (E70) | 0.20 | 0.68 | 0.50 | - |
| blur | hold + restart + verify (recommended) | 0.20 | 0.67 | 0.39 | - |
| blur | hold + restart + strict verify | 0.20 | 0.67 | 0.39 | - |
| blur | restart without monitor gate | 0.21 | 0.98 | 0.33 | - |
| front camera off | per-frame registration | 0.22 | 0.54 | 0.69 | 0.66 |
| front camera off | tracked (E61) | 0.21 | 0.45 | 0.46 | 0.32 |
| front camera off | tracked + hold (E67) | 0.20 | 0.47 | 0.57 | 0.41 |
| front camera off | hold + restart (E70) | 0.20 | 0.39 | 0.46 | 0.40 |
| front camera off | hold + restart + verify (recommended) | 0.20 | 0.37 | 0.36 | 0.31 |
| front camera off | hold + restart + strict verify | 0.20 | 0.37 | 0.36 | 0.31 |
| front camera off | restart without monitor gate | 0.20 | 0.36 | 0.35 | 0.31 |

**Restart events (all four failure types together; events as above)**

| event | restarts of 'hold + restart' (before / during / after the failure) | restarts and verifications of 'hold + restart + verify' | restarts of the ungated restart (before / during / after) |
|---|---|---|---|
| no jump | 0 / 10 / 4 | 0 / 21 / 34 | 11 / 164 / 36 |
| 3 m before | 32 / 15 / 2 | 32 / 26 / 27 | 42 / 163 / 25 |
| 3 m during | 0 / 39 / 15 | 0 / 47 / 34 | 11 / 204 / 25 |
| 6 m during | 0 / 68 / 29 | 0 / 73 / 35 | 11 / 223 / 31 |

## Result images

![Lateral error of the registered map route before, during and after a sensor failure (30-70 % of every scene; mean of night, three forward cameras off, blur and front camera off) for four event sequences.](E71_e71_lines.png)
*Lateral error of the registered map route before, during and after a sensor failure (30-70 % of every scene; mean of night, three forward cameras off, blur and front camera off) for four event sequences.*

![Registration offset error over time in two TEST scenes under the night failure (red band) with a 3 m jump in the middle of the failure (dashed line); triangles mark restarts / verifications. Left: the scene with the median, right: the scene with the largest error of 'hold + restart + verify' in the first 1.5 s after the sensors return (fixed rule, not hand-picked). Below: the monitor p(fail); in the right scene it stays above 0.5 for several seconds on clean frames (false alarms), so the held registration is not corrected until it drops.](E71_e71_trace.png)
*Registration offset error over time in two TEST scenes under the night failure (red band) with a 3 m jump in the middle of the failure (dashed line); triangles mark restarts / verifications. Left: the scene with the median, right: the scene with the largest error of 'hold + restart + verify' in the first 1.5 s after the sensors return (fixed rule, not hand-picked). Below: the monitor p(fail); in the right scene it stays above 0.5 for several seconds on clean frames (false alarms), so the held registration is not corrected until it drops.*

## Findings
- The combined rule costs nothing and repairs both kinds of event. Without a jump 'hold + restart' and 'hold + restart + verify' are as good as the hold alone (lateral error 0.20 / 0.38 / 0.36 / 0.34 m before / during / 0-1.5 s / 1.5-3 s after the failure for the hold alone and for hold + restart, 0.20 / 0.38 / 0.34 / 0.33 m with the verification; floor 0.21 / 0.25 / 0.31 / 0.28 m): the restart condition (monitor healthy AND low score) does not fire spuriously (10 restarts during the 1,176 failure frames of the four failure types, 4 after). With a 3 m jump BEFORE the failure (healthy sensors) the restart of E70 repairs the registration before the failure begins (0.25 m against 0.34 m for the hold alone and 0.29 m for the plain tracker; the ungated restart reaches 0.20 m, the floor level) and the repair carries through the failure (0.50 m during, 0.40 / 0.38 m after, against 0.61 / 0.52 / 0.42 m).
- A jump while the sensors are blind cannot be seen by the perception; it is repaired when the sensors return. During the failure the tracked variants are off by the jump for the second half of it (3 m jump: 0.56-0.62 m lateral error and 1.5-1.6 m offset error, averaged over the whole failure, against 0.25 m for the true registration; 6 m jump: 0.79-0.99 m and 2.3-2.8 m). After the return the restart and the verification take the registration back: for the 3 m jump the lateral error in the first 1.5 s is 0.66 m for the hold alone, 0.51 m with the restart and 0.43 m with restart + verify (0.37 m for the ungated restart, floor 0.31 m), and 0.51 / 0.41 / 0.38 m (0.32 m) in the next 1.5 s; for the 6 m jump 1.25 / 0.56 / 0.56 m (0.37 m) and 0.94 / 0.44 / 0.44 m (0.32 m). The verification matters for the small jump (0.51 -> 0.43 m, offset error 1.22 -> 0.99 m in the first 1.5 s; paired scene bootstrap over the 21 scenes: -0.07 m [-0.15, -0.02] against hold + restart, and -0.23 m [-0.37, -0.11] for hold + restart + verify against the hold alone; for the 6 m jump -0.69 m [-1.07, -0.35] against the hold alone), where the +-1 m window still scores acceptably on a registration that is 3 m off and the restart rarely fires (hold + restart restarts within 1.5 s of the return in 12 of the 84 scene runs); for 6 m the restart alone does the work. The per-frame registration (+-3 m) stays far behind (0.69 / 0.66 m for 3 m, 0.95 / 0.87 m for 6 m), and so does the plain tracker in the first 1.5 s (0.54 m for 3 m, 0.89 m for 6 m; in the next 1.5 s 0.40 m against 0.38 m for the recommended rule at 3 m and 0.64 m against 0.44 m at 6 m).
- The monitor gate is what keeps the restart from hurting during the failure. A restart on a low score without the gate recovers fastest after the return (0.37 / 0.32 m for the 3 m jump, within 0.06 m of the floor; the recommended rule is 0.06 m [-0.02, +0.22] higher for 3 m and 0.18 m [-0.04, +0.51] higher for 6 m in the first 1.5 s, i.e. not significantly with 21 scenes) but it also restarts on degraded graphs: 164 restarts in the 1,176 failure frames without any jump (the gated rule: 10), and the error during the failure doubles (0.74 m against 0.38 m without a jump, a paired scene difference of +0.36 m [+0.20, +0.54]; offset error 2.23 m against 0.80 m for the hold alone and 0.89 m for the recommended rule). By failure type (3 m jump during the failure) night costs 1.17 m against 0.73 m and blur 0.98 m against 0.67 m; for the mild failures, where the perception still works (three forward cameras off, front camera off), the ungated restart is as good as the gated rule (0.42 and 0.36 m against 0.47 and 0.37 m during the failure) - consistent with E67: the gate matters exactly when the perception is wrong. With a 6 m jump during the failure the ungated restart is no longer worse during the failure (0.79 m, the same as the gated rule), presumably because it can also repair the blind jump from the partly informative degraded frames.
- The remaining gap to the ungated restart after the return comes from one scene for the 3 m jump and from a few scenes for 6 m. For the 3 m jump 20 of the 21 scenes recover as fast as with the ungated restart; in one scene (TEST scene #21 of the list; right panel of the time figure) the monitor rates 92 % of the clean frames after the return as failed (p(fail) mostly 0.6-0.8; also 42 % of its frames before the failure), the held registration stays 3-4.7 m off and the lateral error stays at 1.9 m in the first 1.5 s (0.35 m for the ungated restart). Without that scene the lateral error after the return is 0.36 / 0.31 m against 0.37 / 0.31 m for the ungated restart (3 m jump) and 0.42 / 0.32 m against 0.37 / 0.32 m (6 m jump; there scenes #10 and #7 lag as well, 1.4 and 0.9 m against 0.21 and 0.57 m in the first 1.5 s). This is a post-hoc diagnosis, not a result: it shows the failure mode of a gate that keeps holding on a false alarm. Over all runs the monitor rates 7 % of the clean frames before the failure and 5 / 7 / 9 / 14 % of the frames 0-1.5 / 1.5-3 / 3-5 / 5 s or more after the return as failed (without scene #21: 5 % and 0 / 2.5 / 5 / 10 %), so a hold is released at once in nearly all scenes.
- Reading for the system: the recommended registration rule is hold while the monitor fires, restart when the monitor says healthy and the score is low, and verify with the wide search when the sensors return after a hold. It is free in the no-jump case, repairs a jump before a failure and shortens the recovery after a blind jump (for the 3 m jump to within about 0.05 m of the true registration after the return, averaged over the 20 scenes other than #21), and it needs no labels except for the monitor of E64. What it cannot do is detect or bridge a jump while the sensors are blind: that needs a second source (odometry or GNSS consistency, an independent map-matching check) - the held registration is wrong by the jump for the rest of the failure, and for a jump beyond the +-8 m search a global re-localisation would be needed. A further variant - a gate that also accepts a high single-frame score and retries the wide search periodically while holding - brought no gain in a debugging run on the night failure (6 m jump: 0.91 / 0.49 m against 0.78 / 0.50 m for 'hold + restart + verify') and was dropped. A stricter acceptance rule (replace only if the wide search scores better than the held registration) gives the same mean errors in all four event sequences (differences of at most 0.01 m); E72 compares the two rules scene by scene.

## Limits
- Synthetic events: an abrupt sensor failure of 30-70 % of every scene, one failure type at a time, and an instantaneous jump of one magnitude (3 or 6 m) in a random direction at a fixed position of the scene; real jumps have structure (along the road after a GNSS fix), may be followed by drift or by further jumps, and may coincide with an outage of the map matching itself. 21 TEST scenes (84 scene runs per event sequence over four failure types, the clean parts of which are identical), so per-failure-type numbers rest on at most 21 scenes and a single scene (#21) changes the mean after the return by about 0.07 m for the 3 m jump and 0.13 m for 6 m. The blur condition perturbs only every second TEST sample, so its frames are 1 s apart: its time bins (0-1.5 s = 3 frames) and the 'four held frames' of the verification are twice as long in seconds.
- The monitor (E64) is a regression fitted on the even-numbered and applied to the odd-numbered frames of the perturbed TEST runs and vice versa (frame-parity cross-fit); the restart threshold is taken from TEST frames without events (label-free, but from TEST data); the verification rule (>= 4 held frames, accept if the wide search scores within 0.05 of the held registration) was developed over several iterations on the same TEST scenes on which it is evaluated here (an earlier acceptance margin, a gate that also accepts high single-frame scores with periodic wide-search retries - discarded - and a strict acceptance - reported as a variant - were tried on the E71 sequences; the E67 / E70 rules were fixed before E71); it can replace a correct held registration by a wrong wide-search estimate in rare scene runs (E72).
- The lateral error is the open-loop proxy of E62 (anchored, 1 / 2 / 3 s), not a closed-loop result; the pose model is the E61 stress model; the CPU times are those of the unoptimised Python implementation (median of 30 frames: 28 ms for the window search, 69 ms for the full +-3 m search, 72 ms for the wide +-8 m search; 95th percentile 42 / 109 / 122 ms).
