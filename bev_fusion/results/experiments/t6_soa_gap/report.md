# t6_soa_gap - why is the system not at the state of the art? (2026-10-30)

Update 2026-11-01: the same 400 frames were scored again for the later chains (run `pitch.maps_lfl`, GPU variant `temp/soa_gap/map_ref_dev.py`, checked against the CPU run of this report: largest difference 0.0002).
On those identical frames the dual-feature chain (two seeds) stands at 94 / 57 - 62 / 31 - 34 / 26 - 31 / 50 - 52 / 72 - 76 % of GaussianCaR's IoU (drivable, walkway, carpark, crossing, lane divider, road divider; both systems
at the 0.5 threshold); the tables are in `../t3_improve_loop/report.md` ("Maps against the SOA, on identical frames"). The text below is the state of 2026-10-30 (one model, `runs/t1_r4_soa`) and stays as written.

Question (Mirko): "Look exactly why performance is not SOA, find out." Method: measure the gap like for like wherever a published system
can be scored with our own metric on our own frames, then decompose it with the evidence that exists (stored records, earlier arms) and
name what is still unmeasured. CPU only (the GPU was shared with a training lane). Scripts: `temp/soa_gap/` (scratch, not a tool).
All numbers: nuScenes VAL (official val scenes), camera cells of the front camera, 0.5 m grid, front field of view <= 60 m, dt 0, one model
snapshot (`runs/t1_r4_soa`, step 7000, EMA weights), no TEST split.

## 1 Maps: GaussianCaR scored with OUR metric on OUR frames

Reference: GaussianCaR (camera + radar BEV map model, 6 cameras, 360 degrees, trained for map segmentation on the official train split;
weights and logits cached by the lane project for 1706 frames, ALL inside 43 of our 82 official val scenes, none in our train scenes).
Its 200 x 200 array was mapped onto our front grid; the axis convention was found by a search over the 8 orientations on 40 frames
(`align_gcar.py`): row = -y (left), column = forward, drivable IoU 0.842 against 0.547 for the next orientation, so the mapping is
unambiguous; a forward offset of +-0.5 m is not resolved (0.840 at 0, 0.842 at +0.94) and +0.5 m was used. Layers: drivable = max(lane,
road_segment); the others directly; no stop line in GaussianCaR. Same `SegMeter`, same mask (camera cells), same tolerances; threshold
0.5 (best of 0.3 / 0.5 / 0.7 changes ours by <= 0.04, GaussianCaR by <= 0.09 on the dividers). 400 of the 1620 usable frames (every 4th,
`map_ref.py`; the same model scores 0.719 drivable on them against 0.698 on all 3109 frames, so the subset is representative).

| layer (IoU at tolerance) | prior map (null) | ours (t1_r4_soa) | GaussianCaR | ours / SOA |
|---|---|---|---|---|
| drivable area, 0 m | 0.445 | 0.719 | 0.845 | 0.85 |
| walkway, 0 m | 0.000 | 0.256 | 0.594 | 0.43 |
| carpark, 0 m | 0.000 | 0.273 | 0.659 | 0.41 |
| pedestrian crossing, 0 m | 0.000 | 0.092 | 0.536 | 0.17 |
| lane divider, 0.5 m | 0.000 | 0.294 | 0.691 | 0.42 |
| road divider, 0.5 m | 0.000 | 0.351 | 0.634 | 0.55 |

By range (IoU, ours / GaussianCaR): drivable 0-20 m 0.869 / 0.940, 20-40 m 0.747 / 0.888, 40-60 m 0.643 / 0.774; walkway 0.284 / 0.710,
0.279 / 0.678, 0.232 / 0.516; lane divider (0.5 m) 0.368 / 0.808, 0.292 / 0.772, 0.262 / 0.582.

Readings (measured):
1. The map gap is large and real: ours reaches 17-55 % of the SOA IoU on everything except the drivable area (85 %).
2. It is NOT a far-range problem: at 0-20 m, where the front camera sees best and the ground-plane geometry is exact, walkway is
   0.284 against 0.710 and the lane divider 0.368 against 0.808.
3. It is NOT a threshold or calibration problem (best threshold per layer moves ours by <= 0.04).
4. It is NOT overfitting: our TRAIN IoU is already far below the SOA's VAL IoU (drivable 0.711, walkway 0.282, pedestrian crossing 0.140
   on 800 train samples of 428 scenes), and the area loss of train and val is the same (1.034 / 1.077), while detection does overfit
   (objects loss 9.9 / 11.2, mAP 0.448 / 0.309). The maps UNDERFIT: the model cannot even fit the training maps.
5. It is NOT a shortage of capacity in the BEV trunk: the trunk, fusion and temporal blocks were swapped eight times (t3, 3000 steps)
   with ties on walkway and dividers inside the band, and a stride-1 detail path gave nothing.

## 2 Detection

Front sector like for like (t4; DualBEV = the pre-trained SOA camera detector whose image heads we use, run with the front camera only):
mAP2 (vehicle + pedestrian) DualBEV 0.250, ours camera-only 0.242 (tie), ours camera + radar 0.375 / 0.359 (two seeds, 3000 steps), 0.400
(7000 x 2). So the camera path equals the SOA camera path and the radar adds +0.13. A SOA CAMERA + RADAR detector was never run on our frames
(CRN's checkpoint is on disk); the published numbers (CRN 49.0 mAP / 56.0 NDS) are 10 classes, 360 degrees, six cameras and are not
comparable with ours.

Where the AP is lost (t1_r4_soa, 3109 frames, `det_breakdown.py`; the baseline row reproduces the stored vehicle AP@2 0.627 and the headline
mAP 0.309). Oracles applied in order on the stored detections; mean AP over the centre distances 0.5 / 1 / 2 / 4 m:

| class | as evaluated | + perfect centre (within 4 m) | + false positives removed | + misses added |
|---|---|---|---|---|
| vehicle | 0.521 | 0.708 (+0.187) | 0.933 (+0.225) | 1.000 (+0.067) |
| pedestrian | 0.202 | 0.351 (+0.149) | 0.733 (+0.382) | 1.000 (+0.267) |
| two-wheeler | 0.203 | 0.326 (+0.123) | 0.689 (+0.363) | 1.000 (+0.311) |

Recall at the operating point (score >= 0.3, same-group detection within 2 m): vehicles 0.82 / 0.66 / 0.40 at 0-20 / 20-40 / 40-60 m,
pedestrians 0.80 / 0.39 / 0.12; at 40-60 m 37 % of the vehicles and 49 % of the pedestrians have a detection within 2 m with a score below
0.3 (a detection exists, its confidence is low), and 90 % / 76 % have one within 4 m. Motion does not matter for vehicles (moving 0.70, static
0.55 recall; the static ones include the long and far ones); length >= 6 m (trucks, buses) recall 0.36.

What the confident detections are (score >= 0.3 in the front field of view, 10395 of them, `fp_breakdown.py`; each classified against all GT
boxes of the frame including the ones the metric ignores because they carry no lidar / radar point):

| class of the detection | count | share | 0-20 / 20-40 / 40-60 m |
|---|---|---|---|
| same-group observed GT within 2 m (true positives and duplicates of them) | 8206 | 78.9 % | 2313 / 4126 / 1767 |
| background: no GT within 4 m | 1127 | 10.8 % | 123 / 422 / 582 |
| 2-4 m from a same-group GT (localisation or duplicate) | 785 | 7.6 % | 134 / 420 / 231 |
| on an unobserved GT box of the same group | 198 | 1.9 % | 2 / 55 / 141 |
| on an observed GT box of another group (class confusion) | 79 | 0.8 % | 13 / 50 / 16 |

Precision of the confident detections falls with range: 0.89 (0-20 m), 0.81 (20-40 m), 0.64 (40-60 m); at 40-60 m 25 % of them are background.
Counting the unobserved boxes as GT would LOWER the mAP (0.309 -> 0.286), so the metric's rule of ignoring them does not hurt us, and class
confusion (0.8 %) is negligible. So the false positives that cost AP are background detections that grow with range, plus detections 2-4 m off
(duplicates and imprecise centres) - not the evaluation convention and not the class grouping. Caveat: 54 of 3109 frames (1.7 %) have an observed
GT list that differs from the stored record (the GT cache holds files of more than one generator signature and the first file of a token was
used); the percentages are therefore good to about a percentage point.

## 3 Attribution

What is measured and what is only a candidate. "Strong" = measured on this system with a control; "candidate" = consistent with
the numbers, not yet isolated by an experiment.

### 3.1 Maps (the large gap: 17-85 % of the SOA IoU)

| # | factor | verdict | evidence |
|---|---|---|---|
| M1 | far-range depth / geometry | NOT the main cause (strong) | gap is as large at 0-20 m (walkway 0.28 / 0.71, lane divider 0.37 / 0.81) |
| M2 | threshold / calibration | NOT the cause (strong) | best threshold per layer moves ours <= 0.04 |
| M3 | too little data (overfitting) | not the first limit for the big areas (strong), partly for thin and rare classes | train ~ val for drivable (0.711 / 0.698) and the area loss (1.034 / 1.077); train > val for dividers 0.346 / 0.274, carpark 0.296 / 0.162; our TRAIN IoU is below the SOA's VAL IoU on every layer |
| M4 | BEV trunk, fusion, temporal block | NOT the lever (strong) | the block swaps of t3 (gate, CMX, ConvGRU, no temporal block, pillars, adapter, z-sum) tie within the band 0.054 on walkway and dividers; a random-init trunk and a lift without depth are marginally WORSE, nothing is better; the stride-1 detail path (t1_r4_soa) and the corrected FPN order (t4) give nothing at 3000-7000 steps |
| M5 | label quality | not a relative cause (strong) | nuScenes divider layers miss visible paint (t5: 0 m lost by our query), but the SOA scores 0.69 on the same labels |
| M6 | frozen, detection-trained, stride-16 image features (80-channel context) | main candidate | the SOA recipes (BEVFusion camera-only 81.7 drivable / 46.4 divider, GaussianCaR) use stride-8 features trained with the map loss; ours were trained for detection and are frozen (the camera BEV only needs vehicles); t5 probe v1 is INCONCLUSIVE (its RGB control failed); v2 passes its control but is a TIE (stride-16 context F1 0.450, prior 0.437, RGB CNN from scratch 0.488, only the stride-4 stage is better, +0.043 [0.024, 0.061]): a low-resolution probe that cannot rank the features, neither supporting nor refuting M6; the pictures show blurred bands; not isolated. New evidence (t7, another chat, one seed, 2026-10-31): a frozen stride-4 ResNet branch lifted into the BEV lifts the lane divider by +0.065 and the road divider by +0.125 over `ref_s0` but leaves walkway (+0.009), carpark (+0.024) and crossing (-0.002) where they are: resolution is part of it for thin lines, the semantic classes need more than resolution |
| M7 | detection dominates the shared trunk | MEASURED, SMALL (about a tenth of the gap) | gradient norm at the BEV feature at step 7000: objects 0.111, lines 0.012, areas 0.005 (9-22 x). The maps-only run (objects, motion and radar aux off, `experiment.md` section 3, 2026-10-31) gains drivable +0.015, walkway +0.037, carpark +0.040, crossing +0.040, lane divider +0.047, road divider +0.051 over `ref_s0` - one-signed, but inside the band 0.054 and below the gain of doubling the steps (+0.054): 7 - 12 % of the gap to the SOA. R2 of the card |
| M8 | training length | contributes, does not close it | dividers 0.181 -> 0.273 and walkway 0.191 -> 0.246 from 3000 to 7000 x 2 steps (still rising); the SOA trains 20-40 epochs on 700 scenes, we ran 2.6 epochs on 428 |

### 3.2 Detection

| # | finding | evidence |
|---|---|---|
| D1 | The camera path is not behind the SOA camera detector. | front sector, same frames and metric: DualBEV 0.250, ours camera-only 0.242 (tie), camera + radar 0.375 / 0.359 / 0.400 |
| D2 | There is no like-for-like SOA camera + radar number. | CRN is on disk (r50 checkpoint, MIT) but was never run on our frames; the published 49.0 mAP is 10 classes, 360 degrees, six cameras |
| D3 | The AP is lost to centre precision and false positives before misses. | oracles above: vehicle 0.187 / 0.225 / 0.067; the false positives are background detections that grow with range (precision of confident detections 0.89 / 0.81 / 0.64 at 0-20 / 20-40 / 40-60 m), not unobserved boxes (1.9 %) or class confusion (0.8 %) |
| D4 | The far objects are mostly detected but with low confidence. | at 40-60 m a detection within 2 m exists for 37 % (vehicle) / 49 % (pedestrian) of the GT below score 0.3 |
| D5 | Detection overfits strongly; the big map areas do not (thin and rare map classes a little). | mAP 0.448 train / 0.309 val; objects loss 9.9 / 11.2; drivable 0.711 / 0.698 |
| D6 | The velocity is not learned. | AVE 1.82 m/s against 2.40 for predicting zero; moving vehicles predicted 2.7 against 7.6 m/s (observability review) |

### 3.3 What would settle the open candidates (ranked by information per cost)

1. DONE 2026-10-31 - maps-only run (areas + lines, objects and motion weight 0, same steps, same features): the maps gain 0.015 - 0.051, 7 - 12 %
   of the gap: the cause is not the task balance, it is upstream (M6 features, M8 budget). Result in `experiment.md` section 3.
2. Bridge test: our trunk + semantic head trained on the cached GaussianCaR decoder features of the same frames: if it reaches the SOA level,
   everything downstream of the features is fine and the gap is the image side. About 1-2 h GPU plus a data loader for the 54 GB cache.
3. A SOA camera + radar detector (CRN r50, front camera only, the 4-key-frame input) scored on the same 1000 frames: turns D2 into a number.
   Half a day of porting, 30 min GPU.
4. Map-supervised stride-8 image features (plan 2.7, the image-encoder chat is distilling them now): the direct test of M6.
5. Longer training on all 700 scenes with the existing feature shards (M8, M3).

## 4 Question set: why does GaussianCaR win the maps? (analytic; `temp/soa_gap/analytic_questions.py`, `steps_curve.py`)

Arithmetic on the nuScenes camera model (CAM_FRONT f = 1266 px, height 1.51 m), on our image pipeline (704 wide, 256 rows, stride 16, 80
channels) and on GaussianCaR's (448 x 800, stride 8, 128 channels, from the paper note), plus the measured tolerance curves.

| # | question | analytic answer | verdict |
|---|---|---|---|
| Q1 | Does the SOA push more image information into the BEV? | ours 16 x 44 = 704 feature cells x 80 channels = 56,320 values; GaussianCaR 56 x 100 = 5,600 cells x 128 = 716,800: 12.7 x more values, 8 x more cells; per BEV cell of our front grid 4.5 against 57 values | YES, an order of magnitude |
| Q2 | Can a stride-16 cell resolve the ground structures? | one feature cell on the ground at 20 m: ours 0.57 m wide and 7.9 m deep, SOA 0.25 m and 3.4 m; at 40 m ours 1.15 m x 35.5 m, SOA 0.51 m x 13.8 m; a 15 cm marking is 4.2 px at 20 m (0.26 of our cell, 0.59 of the SOA cell). The down-range extent is the real problem: a crossing (stripes 0.5 m wide, 1 m period, 3-4 m deep) or a stop line fills less than one of our cells | YES for crossings, stop lines, divider ends; lateral position of long dividers survives only through sub-cell codes in the features |
| Q3 | Which loss shaped the image features? | ours: detection and lidar depth (DualBEV heads, frozen); SOA: the map loss through the whole backbone. Linear probe on OUR lifted features (t3): drivable 0.577 (prior 0.459), walkway 0.200 (0.121), lane dividers 0.031 (0.018) = dividers are not linearly readable at all | YES, strongest candidate; consistent with Q1-Q2 |
| Q4 | Does the SOA train longer? | 40 epochs x 28,130 frames = 1.13 M sample passes against 42 k - 126 k (2.6 - 7.7 epochs of 16,289) = 9 - 27 x more | YES (see section 5) |
| Q5 | Is the deficit one of POSITION (blur) or of PRESENCE (the class is not recognised)? | ratio ours / SOA at tolerance 0 / 0.5 / 1 m: drivable 0.85 / 0.86 / 0.88, walkway 0.43 / 0.45 / 0.48, carpark 0.41 / 0.41 / 0.44, crossing 0.17 / 0.18 / 0.19, lane divider 0.46 / 0.42 / 0.49; a blur would close toward 1 with the tolerance, it does not; at 1 m the missing IoU is still 0.41 (walkway), 0.51 (carpark), 0.59 (crossing), 0.43 (lane divider) | PRESENCE: the network does not recognise the classes in many places; boundary precision is secondary |
| Q6 | Does the SOA see more of our cells (side cameras)? | 26.5 % of the front-cone cells are also inside a side camera's field (the outer edges) | minor: cannot give 2 - 5 x |
| Q7 | Does the radar explain it? | our own ablation: camera-only -> camera + radar adds drivable +0.03, walkway +0.08, dividers +0.02 (t3, 3000 steps); static layers are camera information | NO |
| Q8 | Is the BEV trunk or head the limit? | eight block swaps tie (M4); but a 26 M-parameter trunk lifts walkway only from the linear-probe 0.200 to 0.244 (different frame sets, indicative): it extracts little the features do not hold | NO (the information is upstream) |
| Q9 | Range / depth? | gap equal or larger at 0-20 m (section 1) | NO |
| Q10 | Labels, threshold, scenes seen in training? | same labels; best-threshold changes <= 0.04; GaussianCaR scored only on official val scenes | NO |

Reading: the deficit is mainly one of PRESENCE (Q5) with an information bottleneck upstream (Q1-Q3), compounded by a much smaller training budget
(Q4). The SOA's advantage is not in the BEV network, which we replaced eight times without gain.

## 5 Why do the maps underfit, and does more training help? (`steps_curve.py`)

The same configuration trained for 3000, 6000 and 7000 x 2 steps (VAL-1000, camera cells, dt 0):

| steps (accum) | drivable | walkway | carpark | crossing | lane divider (0.5 m) | road divider (0.5 m) |
|---|---|---|---|---|---|---|
| 3000 (ref_s0) | 0.666 | 0.191 | 0.106 | 0.002 | 0.181 | 0.220 |
| 6000 (ref_long) | 0.689 | 0.225 | 0.155 | 0.057 | 0.234 | 0.297 |
| 7000 x 2 (r4_soa) | 0.698 | 0.246 | 0.160 | 0.093 | 0.273 | 0.320 |
| GaussianCaR (400 frames) | 0.845 | 0.594 | 0.659 | 0.536 | 0.691 | 0.634 |

Every layer still rises, so the model is not at its limit: more training helps. But the rate does not carry it to the SOA. Gain per doubling of
the steps, first doubling (3000 -> 6000): drivable +0.023, walkway +0.034, carpark +0.049, crossing +0.055, lane divider +0.053, road divider
+0.077; it shrinks in the next step (drivable +0.009 per 1.2 doublings). Even at the optimistic first-doubling rate the remaining gap to the SOA
needs 4 (road divider) to 10 (walkway, carpark) more doublings = 16 - 1000 x the compute; the real curves are concave, so these are lower
bounds. Longer training alone does not close the gap of this setup.

Why the maps underfit (ranked; the first two are supported by measurements, the rest are mechanisms the numbers allow):
1. The features hold little of the map classes (Q1-Q3): with frozen detection-trained stride-16 features the trained model cannot separate the
   training maps either (train IoU walkway 0.282, crossing 0.140). Optimisation cannot create information the features lack.
2. The rare thin classes take off late: crossing 0.002 -> 0.057 -> 0.093 and carpark 0.106 -> 0.155 -> 0.160. The positive cells are roughly
   0.5 - 3 % of the evaluated cells (n_gt of the 400-frame run over about 2.6 M cells, my estimate of the cell count), and a BCE + soft-IoU
   loss sits near "predict nothing" for a long time on such classes (focal losses or class weights shorten that plateau - a mechanism, not
   tested here; GaussianCaR trains 9 - 27 x longer).
3. Task competition: at the BEV feature the gradient of the objects is 9 - 22 x that of lines and areas (step 7000: 0.111 / 0.012 / 0.005) although
   the Kendall weights down-weight the objects. TESTED (maps-only run, 2026-10-31): it is real but small - removing the objects and the motion
   head lifts the thin layers by 0.037 - 0.051 and the drivable area by 0.015 (about a tenth of the gap), so it is a minor factor. The earlier
   observation that t1_r4_soa fits the TRAIN lane dividers worse (0.346) than t1_r0 (0.446) despite more steps is therefore not explained by
   this alone (that run changed several things at once).
4. Data: 428 of 700 scenes; thin classes overfit a little (0.346 train / 0.274 val), the large areas not.

