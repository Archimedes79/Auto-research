# E72: Intermittent failures: three sensor-failure bursts per scene, with and without a jump of the localisation

Status: completed

**Question:** E67 and E71 use ONE long failure in the middle of every scene. Real failures are often intermittent - glare, a tunnel, a dirty lens, a flapping connection - so the registration is held, released, held again. Does the recommended rule (hold while the monitor fires, restart when healthy and the score is low, verify when the sensors return) survive repeated bursts without degrading, does the verification at every return hurt when nothing happened, and is a jump of the localisation late in one burst repaired by the next return and gone in the next burst?

## Setup
- Training / test diagnosis: nothing is trained; the same code, monitor (E64, cross-fitted over the frame parity), thresholds (restart threshold 0.631, wide search +-8 m / +-8 deg) and E61 pose model as in E71 (combo_eval.py with --windows). TEST only: 21 scenes; the clean frames before the first burst are the seen condition, the bursts the unseen one.
- Scenario: every TEST scene gets THREE failure bursts at 10-25 %, 40-55 % and 70-85 % of its length (about 3 s each in a 20 s scene; the perturbed graphs of E57 - night, three forward cameras off, blur, front camera off - during the bursts, clean graphs in between), 1,323 failure frames in all four failure types. Two event sequences: no jump, and a 3 m jump of the map offset in a random direction at 50 % of the scene, i.e. in the last third of the second burst (the sensors are blind; in a 40-frame scene burst 2 covers frames 16-21 and the jump is at frame 20). The bursts are at fixed positions, not random. The blur condition perturbs only every second TEST sample (20 frames per scene, 1 s apart), so its bursts are three frames = 3 s long as well.
- Methods and metrics as in E71: per-frame registration, tracked registration, tracked + monitor-gated hold, hold + restart, hold + restart + verify (the recommended rule), a 'strict verify' variant (the held registration is replaced only if the wide search scores better than it), a restart on a low score WITHOUT the monitor gate and the true registration (floor). Anchored lateral error of the registered map route (E62) and registration offset error; 'after a burst' = time since the end of the most recent burst. Lateral errors exist only for frames with 3 s of future motion, so the last 3 s of each scene (the return after burst 3) are not evaluated.

![architecture](architecture.png)

## Numbers

**Lateral error (m, anchored) of the registered map route by phase (mean over four failure types, the 1.5-3 s column over three because the blur scenes have no frames there; 21 TEST scenes; three bursts per scene at 10-25 %, 40-55 % and 70-85 % of the scene). Event: no jump / a 3 m jump in the last third of the second burst (at 50 %, the sensors are blind). 'After a burst' = time since the end of the most recent burst**

| event | method | lateral error (m): before the first burst | lateral error (m): during the bursts | lateral error (m): after a burst: 0-1.5 s | lateral error (m): after a burst: 1.5-3 s |
|---|---|---|---|---|---|
| no jump | per-frame registration | 0.21 | 0.61 | 0.33 | 0.40 |
| no jump | tracked (E61) | 0.21 | 0.45 | 0.30 | 0.34 |
| no jump | tracked + hold (E67) | 0.21 | 0.34 | 0.29 | 0.32 |
| no jump | hold + restart (E70) | 0.21 | 0.34 | 0.29 | 0.32 |
| no jump | hold + restart + verify (recommended) | 0.21 | 0.34 | 0.29 | 0.32 |
| no jump | hold + restart + strict verify | 0.21 | 0.34 | 0.29 | 0.32 |
| no jump | restart without monitor gate | 0.21 | 0.73 | 0.29 | 0.31 |
| no jump | true registration (floor) | 0.21 | 0.23 | 0.26 | 0.29 |
| 3 m jump in burst 2 | per-frame registration | 0.21 | 0.69 | 0.47 | 0.49 |
| 3 m jump in burst 2 | tracked (E61) | 0.21 | 0.52 | 0.48 | 0.43 |
| 3 m jump in burst 2 | tracked + hold (E67) | 0.21 | 0.49 | 0.49 | 0.47 |
| 3 m jump in burst 2 | hold + restart (E70) | 0.21 | 0.42 | 0.35 | 0.36 |
| 3 m jump in burst 2 | hold + restart + verify (recommended) | 0.21 | 0.41 | 0.34 | 0.34 |
| 3 m jump in burst 2 | hold + restart + strict verify | 0.21 | 0.42 | 0.34 | 0.36 |
| 3 m jump in burst 2 | restart without monitor gate | 0.21 | 0.72 | 0.29 | 0.32 |
| 3 m jump in burst 2 | true registration (floor) | 0.21 | 0.23 | 0.26 | 0.29 |

**Registration offset error (m) at the vehicle by phase (mean over four failure types; events as above)**

| event | method | offset error (m): before | offset error (m): during | offset error (m): after 0-1.5 s | offset error (m): after 1.5-3 s |
|---|---|---|---|---|---|
| no jump | per-frame registration | 0.74 | 1.84 | 1.01 | 1.14 |
| no jump | tracked (E61) | 0.73 | 1.09 | 0.67 | 0.72 |
| no jump | tracked + hold (E67) | 0.72 | 0.80 | 0.60 | 0.64 |
| no jump | hold + restart (E70) | 0.72 | 0.82 | 0.62 | 0.64 |
| no jump | hold + restart + verify (recommended) | 0.72 | 0.83 | 0.65 | 0.65 |
| no jump | hold + restart + strict verify | 0.72 | 0.82 | 0.62 | 0.64 |
| no jump | restart without monitor gate | 0.74 | 2.16 | 0.76 | 0.71 |
| no jump | true registration (floor) | 0.00 | 0.00 | 0.00 | 0.00 |
| 3 m jump in burst 2 | per-frame registration | 0.74 | 1.95 | 1.14 | 1.19 |
| 3 m jump in burst 2 | tracked (E61) | 0.73 | 1.35 | 1.07 | 0.95 |
| 3 m jump in burst 2 | tracked + hold (E67) | 0.72 | 1.28 | 1.20 | 1.05 |
| 3 m jump in burst 2 | hold + restart (E70) | 0.72 | 1.15 | 0.92 | 0.75 |
| 3 m jump in burst 2 | hold + restart + verify (recommended) | 0.72 | 1.15 | 0.93 | 0.73 |
| 3 m jump in burst 2 | hold + restart + strict verify | 0.72 | 1.15 | 0.91 | 0.75 |
| 3 m jump in burst 2 | restart without monitor gate | 0.74 | 2.22 | 0.91 | 0.78 |
| 3 m jump in burst 2 | true registration (floor) | 0.00 | 0.00 | 0.00 | 0.00 |

**Per failure type (3 m jump in the second burst)**

| condition | method | lateral error (m): before the first burst | lateral error (m): during the bursts | lateral error (m): after a burst: 0-1.5 s | lateral error (m): after a burst: 1.5-3 s |
|---|---|---|---|---|---|
| night | per-frame registration | 0.20 | 0.93 | 0.46 | 0.49 |
| night | tracked (E61) | 0.21 | 0.62 | 0.46 | 0.39 |
| night | tracked + hold (E67) | 0.20 | 0.54 | 0.50 | 0.49 |
| night | hold + restart (E70) | 0.20 | 0.49 | 0.38 | 0.40 |
| night | hold + restart + verify (recommended) | 0.20 | 0.47 | 0.35 | 0.37 |
| night | hold + restart + strict verify | 0.20 | 0.49 | 0.36 | 0.39 |
| night | restart without monitor gate | 0.21 | 1.24 | 0.32 | 0.34 |
| 3 forward cameras off | per-frame registration | 0.20 | 0.57 | 0.46 | 0.49 |
| 3 forward cameras off | tracked (E61) | 0.21 | 0.48 | 0.52 | 0.49 |
| 3 forward cameras off | tracked + hold (E67) | 0.20 | 0.47 | 0.48 | 0.47 |
| 3 forward cameras off | hold + restart (E70) | 0.20 | 0.39 | 0.31 | 0.32 |
| 3 forward cameras off | hold + restart + verify (recommended) | 0.20 | 0.40 | 0.31 | 0.32 |
| 3 forward cameras off | hold + restart + strict verify | 0.20 | 0.39 | 0.32 | 0.32 |
| 3 forward cameras off | restart without monitor gate | 0.21 | 0.42 | 0.28 | 0.30 |
| blur | per-frame registration | 0.23 | 0.79 | 0.51 | - |
| blur | tracked (E61) | 0.23 | 0.57 | 0.50 | - |
| blur | tracked + hold (E67) | 0.24 | 0.52 | 0.52 | - |
| blur | hold + restart (E70) | 0.24 | 0.44 | 0.40 | - |
| blur | hold + restart + verify (recommended) | 0.24 | 0.44 | 0.40 | - |
| blur | hold + restart + strict verify | 0.24 | 0.44 | 0.40 | - |
| blur | restart without monitor gate | 0.23 | 0.89 | 0.28 | - |
| front camera off | per-frame registration | 0.20 | 0.49 | 0.46 | 0.49 |
| front camera off | tracked (E61) | 0.21 | 0.40 | 0.43 | 0.41 |
| front camera off | tracked + hold (E67) | 0.20 | 0.43 | 0.46 | 0.45 |
| front camera off | hold + restart (E70) | 0.20 | 0.38 | 0.31 | 0.35 |
| front camera off | hold + restart + verify (recommended) | 0.20 | 0.33 | 0.30 | 0.32 |
| front camera off | hold + restart + strict verify | 0.20 | 0.38 | 0.31 | 0.35 |
| front camera off | restart without monitor gate | 0.21 | 0.34 | 0.28 | 0.30 |

**Restart events (all four failure types together; events as above)**

| event | restarts of 'hold + restart' (before / during / after the failure) | restarts and verifications of 'hold + restart + verify' | restarts of the ungated restart (before / during / after) |
|---|---|---|---|
| no jump | 0 / 16 / 6 | 0 / 19 / 58 | 3 / 189 / 75 |
| 3 m jump in burst 2 | 0 / 37 / 22 | 0 / 41 / 67 | 3 / 220 / 71 |

**Lateral error (m) per burst (mean over four failure types): during each burst and in the first 1.5 s after it; the 3 m jump happens in the last third of burst 2; after burst 3 the last 3 s of each scene have no future positions and are not evaluated**

| event | method | during burst 1 | 0-1.5 s after burst 1 | during burst 2 | 0-1.5 s after burst 2 | during burst 3 |
|---|---|---|---|---|---|---|
| no jump | tracked + hold (E67) | 0.25 | 0.29 | 0.32 | 0.29 | 0.47 |
| no jump | hold + restart (E70) | 0.25 | 0.29 | 0.33 | 0.29 | 0.46 |
| no jump | hold + restart + verify (recommended) | 0.25 | 0.29 | 0.32 | 0.29 | 0.47 |
| no jump | hold + restart + strict verify | 0.25 | 0.29 | 0.33 | 0.29 | 0.46 |
| no jump | restart without monitor gate | 0.62 | 0.29 | 0.73 | 0.28 | 0.76 |
| no jump | true registration (floor) | 0.19 | 0.28 | 0.20 | 0.24 | 0.30 |
| 3 m jump in burst 2 | tracked + hold (E67) | 0.25 | 0.29 | 0.51 | 0.69 | 0.70 |
| 3 m jump in burst 2 | hold + restart (E70) | 0.25 | 0.29 | 0.47 | 0.41 | 0.55 |
| 3 m jump in burst 2 | hold + restart + verify (recommended) | 0.25 | 0.29 | 0.46 | 0.39 | 0.51 |
| 3 m jump in burst 2 | hold + restart + strict verify | 0.25 | 0.29 | 0.47 | 0.40 | 0.55 |
| 3 m jump in burst 2 | restart without monitor gate | 0.62 | 0.29 | 0.76 | 0.29 | 0.72 |
| 3 m jump in burst 2 | true registration (floor) | 0.19 | 0.28 | 0.20 | 0.24 | 0.30 |

## Result images

![Lateral error of the registered map route before the first burst, during the three sensor-failure bursts and in the 0-1.5 s / 1.5-3 s after a burst (mean of night, three forward cameras off, blur and front camera off), without a jump and with a 3 m jump in the last third of the second burst.](e72_lines.png)
*Lateral error of the registered map route before the first burst, during the three sensor-failure bursts and in the 0-1.5 s / 1.5-3 s after a burst (mean of night, three forward cameras off, blur and front camera off), without a jump and with a 3 m jump in the last third of the second burst.*

![Registration offset error over time in two TEST scenes with three night-failure bursts (red bands) and a 3 m jump in the last third of the second burst (dashed line); triangles mark restarts / verifications. Left: the scene with the median, right: the scene with the largest error of 'hold + restart + verify' in the first 1.5 s after the bursts (fixed rule, not hand-picked). Below: the monitor p(fail).](e72_trace.png)
*Registration offset error over time in two TEST scenes with three night-failure bursts (red bands) and a 3 m jump in the last third of the second burst (dashed line); triangles mark restarts / verifications. Left: the scene with the median, right: the scene with the largest error of 'hold + restart + verify' in the first 1.5 s after the bursts (fixed rule, not hand-picked). Below: the monitor p(fail).*

## Findings
- Repeated holds and returns cost nothing when nothing happens. Without a jump, hold, hold + restart and hold + restart + verify have the same lateral error - 0.21 / 0.34 / 0.29 / 0.32 m before the first burst / during the bursts / 0-1.5 s / 1.5-3 s after a burst against 0.21 / 0.23 / 0.26 / 0.29 m for the true registration - and are better than the tracker updated at every frame (0.45 / 0.30 / 0.34 m) and the per-frame registration (0.61 / 0.33 / 0.40 m). Restarts and verifications replaced the held registration 58 times after a burst and 19 times inside the bursts (hold + restart alone: 6 and 16 restarts) with no change in lateral error; the registration offset error is at most 0.05 m higher after a return (0.65 against 0.60 m in the first 1.5 s), the price of replacing the held estimate by a wide-search estimate that is a little noisier. The restart condition itself fired only 16 times inside the 1,323 burst frames without a jump (the gate keeps it off degraded graphs).
- The restart WITHOUT the monitor gate is the one that suffers from repeated bursts: 189 restarts in the 1,323 burst frames (3 / 189 / 75 before / during / after) and a lateral error of 0.73 m during the bursts against 0.34 m for the gated rule (paired scene difference +0.38 m [+0.25, +0.52]), 0.62 / 0.73 / 0.76 m in bursts 1 / 2 / 3 (0.25 / 0.32 / 0.47 m for the gated rule), then 0.29 / 0.31 m after the bursts - the same trade as in E71, now three times: it is good right after a return and bad while the sensors fail. By failure type it costs 1.24 m against 0.47 m at night and 0.89 m against 0.44 m under blur (3 m jump run) and is neutral for the mild failures (front camera off 0.34 against 0.33 m, three forward cameras off 0.42 against 0.40 m).
- A jump late in burst 2 is repaired at the next return and does not carry over. With the 3 m jump the lateral error during the bursts is 0.41 m for hold + restart + verify (0.42 m without the verification, 0.49 m for the hold alone, 0.52 m for the tracker, 0.72 m for the ungated restart; floor 0.23 m), and 0.34 / 0.34 m in the first 1.5 s / 1.5-3 s after a burst against 0.49 / 0.47 m for the hold alone (paired scene difference over the first 3 s after a burst -0.14 m [-0.24, -0.05]; +0.05 m [+0.01, +0.10] against the ungated restart). Per burst (verification rule): 0.25 m in burst 1, 0.46 m in burst 2 (the jump comes in its last third), 0.39 m in the first 1.5 s after it (hold alone 0.69 m, ungated restart 0.29 m, floor 0.24 m) and 0.51 m in burst 3 - about what burst 3 costs without any jump (0.47 m), whereas the hold alone is still 0.23 m worse than without the jump (0.70 m).
- What remains is the gap to the ungated restart right after the return (0.39 against 0.29 m after burst 2): three of the 21 scenes (#6, #21, #13) keep an error of 0.9-1.0 m in the first 1.5 s after the return (0.11 / 0.37 / 0.64 m for the ungated restart) - scene #21 is the scene of E71 in which the monitor keeps rating clean frames as failed. The error DURING the burst in which the jump happens (0.46 m against 0.20 m for the floor in burst 2) is not repairable from the perception, as in E71.
- The verification step is not free of risk. It replaces the held registration by the wide-search estimate whenever that scores within 0.05 of the held one, so it can replace a correct registration by a wrong one (presumably when the wide search finds an equally good fit on a repeated lane structure; the mechanism was not verified): in the scene on the right of the time figure (a night scene, TEST scene #16) the verification after burst 2 jumps to an offset of 4.4 m and the registration stays 1.7-6 m off for the rest of the scene, which hold + restart without it avoids (0.2-1.7 m); the effect on the mean lateral error of this scene run is only +0.09 m. Averaged over scene runs (failure and after phases) the effect is small: over the six event sequences of E71 and E72 (504 scene runs, each one scene and one failure type) the verification changes the mean lateral error by more than 0.1 m in 14 runs - 10 times for the better, 4 times for the worse (the worst case +0.27 m); the 10 improved runs come from 5 scenes (5 of them from scene #21), the 4 worsened ones from 3 scenes; in the no-jump sequences it never changes the error by more than 0.19 m. A stricter acceptance rule ('strict verify' in the tables: replace only if the wide search scores better than the held registration) has the same mean errors as the standard rule in all four E71 sequences (differences of at most 0.01 m) and gives up the small gain of the verification in E72 (after a burst 0.34 / 0.36 m instead of 0.34 / 0.34 m, which is the level of hold + restart; paired scene difference to the standard verification +0.011 m [0.000, 0.030], while the standard verification is 0.014 m [0.002, 0.034] below hold + restart); it changes a scene run by more than 0.1 m in only 6 of the 504 runs - 5 times for the better, once for the worse - compared with hold + restart; compared with the standard verification it is better in 3 and worse in 5 runs. The standard rule stays the recommended one because its net gain is larger (10 improved against 4 worsened runs); the strict rule is the conservative alternative.
- Reading for the system: the recommended rule can be left running on a stream of intermittent failures - no flapping problems showed up in this test (the hold is entered and released up to three times per scene - 3.1 times for the night failure, 0.9-2.0 for the milder types - and restarts are rare: 16 in the 1,323 burst frames, 6 after a burst and none before the first burst, with no change in error; the lateral error in the three bursts, 0.25 / 0.32 / 0.47 m, rises from burst to burst and so does that of the true registration, 0.19 / 0.20 / 0.30 m - the gap grows from 0.06 to 0.17 m, and this test cannot separate the effect of repeated holds from that of the later parts of the scenes), the verification is free in the no-jump case, and a jump that happens while a burst blinds the sensors is repaired at the next return. The cost of using the ungated restart as a shortcut recurs with every burst. What the test does not cover: bursts that are shorter than the monitor's reaction (one or two frames), failures that overlap (night + radar loss), and bursts of random length at random positions.

## Limits
- Synthetic bursts of one type at a time, three per scene at fixed positions, 3 s each (three frames in the blur scenes, whose frames are 1 s apart), with abrupt on / off switching of the perturbed graphs; real intermittent failures have ramps and varying length. One jump size (3 m) at one position (the last third of burst 2). The time bins in seconds assume 0.5 s per frame and are twice as long for blur. 21 TEST scenes, the clean parts of which are the same for all four failure types; per-failure-type numbers rest on at most 21 scenes, and the gaps after the return are carried by a few scenes (three in the 3 m run).
- The monitor (E64) is a regression cross-fitted over the frame parity of the perturbed TEST runs; the restart threshold comes from TEST frames without events (label-free, but TEST data); the verification rule is that of E71 and was developed on the E71 events on the same scenes. The lateral error is the open-loop proxy of E62 and exists only for frames with 3 s of future motion (no evaluation after burst 3); the pose model is the E61 stress model.
