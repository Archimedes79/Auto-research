# Radar grid encoding - ideas for discussion (2026-10-25)

Working document of the radar block (plan.md section 3 is the agreed state; this file collects the ideas until one is chosen). The camera block is fixed: lift PD, V(cell) = c(pixel) * p(depth of the cell | pixel), gathered through a lookup table per calibration (plan.md section 2).

## 0. What a nuScenes radar measurement looks like (measured, `temp/radar/radar_stats.py`, 300 random sweeps per radar, all detections, no filters)
| radar | detections per sweep (median / max) | range 1 / 50 / 99 % | beyond 60 m | azimuth 1 / 99 % |
|---|---|---|---|---|
| FRONT | 125 / 125 | 6.7 / 51.7 / 155 m | 40 % | -61 / +61 deg |
| FRONT_LEFT | 49 / 125 | 7.0 / 49.1 / 209 m | 39 % | -63 / +64 deg |
| FRONT_RIGHT | 40 / 125 | 6.4 / 46.1 / 207 m | 36 % | -63 / +65 deg |
| BACK_LEFT | 125 / 125 | 6.6 / 50.9 / 193 m | 40 % | -59 / +62 deg |
| BACK_RIGHT | 125 / 125 | 6.2 / 49.6 / 193 m | 38 % | -60 / +62 deg |
- the list is capped at 125 detections per sweep and the front / back radars sit at the cap: what the radar reports is already a truncated selection
- z is always 0 (2D radar: no elevation)
- `x_rms`, `y_rms`, `vx_rms` are integer quality codes (19-23, 19-23, 16-20; 0 for about 6 % of the detections), not metres - they enter as categorical embeddings, not as standard deviations; `vy_rms` is constant (3) and carries nothing
- `invalid_state` and `ambig_state` vary a lot (e.g. FRONT: invalid_state 0 for about half of the detections) - kept as features, not used as filters

Doppler, measured (`temp/radar/doppler_stats.py`, 213,276 detections of 400 random sweeps per radar, no filters):
- `vx`, `vy` are NOT only the radial component: their component across the line of sight reaches 25.0 m/s (99 % quantile), so the sensor reports a 2D velocity estimate per detection (the earlier assumption "radial vector" was wrong); the radial part v_r = (vx, vy) . u is computed from them
- the sensor resolves the Doppler ambiguity itself: `ambig_state` 3 (85 % of the detections) stays within raw v_r -28.2..+26.2 m/s, states 0 / 1 within about +-25 m/s; state 2 (1.7 %) reaches -90.4..+79.8 m/s; no folding at a fixed interval is visible in nuScenes - the unambiguous interval of the sensor cannot be read from these data

## 1a. Input encodings of a detection (Mirko, 2026-10-25)
- original sensor ray direction: the line of sight u at the time of the measurement, as a circular value (cos phi, sin phi) - in the sensor frame (where the azimuth noise lives) and in the vehicle frame of the measurement time (where the ego motion and the static Doppler live). After ego-motion compensation of an older sweep the ORIGINAL u is kept: the Doppler was measured along it (v_r = v . u_original), so the static residual and the Doppler information A, b use u_original, not the direction recomputed from the compensated position.
- Doppler as a circular value: a radar measures the radial velocity only modulo its unambiguous interval [-v_max, +v_max); a target faster than v_max folds to the other end, so +v_max and -v_max are neighbours. Encoding the phase psi = pi v_r / v_max as (sin psi, cos psi) gives the network exactly that topology: a folded measurement stays close to its true neighbourhood instead of jumping across the input range. Precision: add harmonics (sin k psi, cos k psi), k = 1, 2, 4, 8 - all with the same period, so the folding topology is kept while small velocity differences stay resolvable.
- v_max is a property of the sensor model (like the intrinsics of a camera: an input, not a constant of the network); for nuScenes it is not visible in the data (above), so: v_max from the sensor specification if known, else a value above the observed range (e.g. 30 m/s for the unambiguous states); the linear (scaled) v_r and an embedding of `ambig_state` are given as well, so the network can rely on the circular code where the sensor is ambiguous and on the linear value where it is not.
- the same circular code for the static residual r_s = wrap(v_r - v_static(u_original)) (wrap into [-v_max, v_max)) and for the tangential component of (vx, vy) (linear, since it is the sensor's own estimate, not a folded measurement).
- the other inputs: range (linear, scaled), RCS, the quality codes x_rms / y_rms / vx_rms and the states dyn_prop / invalid_state / ambig_state as embeddings, dt, sensor id.

### 1b. The input vector per detection: simple, yet complete (Mirko: "simple, easy, yet complete values")
Everything a nuScenes detection carries, each value once, in the frame where it is simple; dropped only what carries nothing (z = 0 always, vy_rms constant, the cluster id) or what is derivable (vx_comp / vy_comp: replaced by the static residual).
| group | values | encoding | dims |
|---|---|---|---|
| where (sensor frame) | range r; azimuth phi_s | r / 60 (linear); cos phi_s, sin phi_s | 3 |
| where (vehicle frame, measurement time) | direction of the line of sight; sensor position | cos phi_v, sin phi_v; x_s / 10, y_s / 10 | 4 |
| radial velocity (measured) | v_r = (vx, vy) . u | circular sin / cos of k pi v_r / v_max, k = 1, 2, 4, 8; plus v_r / v_max linear | 9 |
| tangential velocity (sensor estimate) | v_t = (vx, vy) x u | v_t / v_max linear | 1 |
| static residual (moving or not) | r_s = wrap(v_r - v_static(u)) with v_static from the ego motion | circular sin / cos of k pi r_s / v_max, k = 1, 2; plus linear | 5 |
| strength | RCS (dBsm) | RCS / 20 | 1 |
| age | dt = t_ref - t_measurement | dt / 0.5 s | 1 |
| quality and state (categorical) | x_rms, y_rms, vx_rms codes (0 = missing); dyn_prop; ambig_state; invalid_state | one learned embedding of 2-4 dims each | about 18 |
| total | | | about 42 |
v_max: from the sensor model (section 1a); for nuScenes a value above the observed range of the unambiguous states (30 m/s) until a specification is known. This vector is the input of both options: the point encoder of A, and the per-cell channels of the polar image in B (averaged per cell, plus the count and the sub-cell offsets).

## 1. Mirko's idea: a feature per detection and a decoder around it
f_j = E(r_j, phi_j, data_j) per detection; a decoder g(f_j, x + l, y + m) gives the values of the cells around the detection (l, m in -1..1).

It makes sense - it is the published form of a learned continuous kernel: Deep Parametric Continuous Convolution 2101.06742 (kernel g = MLP of the continuous offset between output cell and input point; output cells may be a dense BEV grid) and ContFuse 2012.10992 (h_i = sum_j MLP(f_j, x_j - x_i); their ablation: removing the geometric offset input hurts significantly). Three refinements:
1. the offset must be continuous, not the integer (l, m): a detection near a cell edge and one at a cell centre would otherwise produce the same pattern (aliasing of 0.5 m). Input: the exact offset from the detection to each cell centre.
2. the offset should be in the sensor's own coordinates: (r_cell - r_j, phi_cell - phi_j) from the calibration, not (dx, dy) - the azimuth noise is then the same along one axis at every range and every mounting, and the same g serves every radar (rig independence).
3. the window must grow with range: 3 x 3 cells of 0.5 m is +-0.75 m; at 50 m one degree of azimuth is already 0.87 m. A window that is fixed in polar terms (+-k range bins, +-k azimuth bins) grows by itself.

With these three it is the "sparse decoder" option A below.

## 2. The asymmetry, and how to remove it
Camera and radar have the same form once written as "measurement feature x learned likelihood of the cell given the measurement":
- camera: V(cell) = c(pixel) * p(depth of the cell | pixel)
- radar (option A): V(cell) = sum_j f_j * w(cell | detection j)
The difference is only where the measurements live: camera measurements sit on a fixed dense grid (the pixels), so every cell can PULL through a fixed table; radar detections are sparse and move, so option A must PUSH (scatter). The symmetric version puts the radar on a fixed dense grid of its own first - its measurement image in polar coordinates - and then uses exactly the camera machinery: sensor image -> backbone -> feature x likelihood -> pulled by every BEV cell through a fixed table per calibration. That is option B.

## 3. Option B: radar sensor image (polar) + backbone + table pull - the symmetric path (recommended)
1. Radar image per sensor: a polar grid in the sensor frame, range 0-64 m x azimuth -66..+66 deg, e.g. 0.5 m x 1 deg = 128 x 132 cells (or 0.25 m x 0.5 deg = 256 x 264). In polar coordinates the radar's uncertainty is stationary: the azimuth error is a fixed number of bins at every range, just as the camera's lateral error is a fixed number of pixels.
2. Rasterise the detections of the last T sweeps (about 0.5 s; each moved to the current sensor pose by the ego motion, radially shifted by v_r * dt) into it, channels per cell (encodings of section 1a, averaged over the detections of the cell): count; sum and max RCS; Doppler v_r as the circular code plus the linear value; static residual as the circular code; the original ray direction (in a polar cell of an older sweep it differs from the cell's own direction); exact sub-cell offsets dr, dphi of the detection (the cell + offset code of FFT-RadNet 2112.10646, so rasterising loses no position); youngest and mean age dt; embeddings of the quality codes and states; Doppler information A, b (in a polar cell the line of sight u is known, so A and b are plain sums).
3. Radar backbone: a small 2D CNN on the polar image (range x azimuth), the same weights for every radar (the deep radar ISM 2305.12409 applies its network per sensor and stays usable on an unseen mounting: 68.7 -> 63.5 mIoU front centre). Its receptive field spreads evidence in polar terms - wide in azimuth, narrow in range - and it sees the neighbours of a detection (a wall of returns, an object's extent), which the per-detection kernel of option A cannot.
4. Output per polar cell: a feature c_rad and a likelihood / occupancy logit p_rad (the radar's counterpart of the depth distribution; optionally a variance per cell as in Weston et al. 1810.08151).
5. Lift: a fixed table per radar, BEV cell (x, y) -> its (r, phi) in that radar's polar image and a valid flag (inside the field of view); V_rad(x, y) = sum over radars of c_rad(r, phi) * p_rad(r, phi), two grid_sample gathers as in lift PD. Radar has no height: the result is one BEV layer (or copied into the height slices with a height prior).
- Basis: deep radar ISM 2305.12409 (detections binned into the sensor's polar grid, a CNN learns the footprint; learned vs geometric Gaussian mixture: P(occupied | occupied) 38.7 vs 15.3 %), Weston et al. 1810.08151 (polar encoder + fixed polar-to-Cartesian bilinear resampling inside the network, per-cell variance), FFT-RadNet 2112.10646 (polar cell + offset code), RadarOcc 2405.14014 (keeping the radar in its own polar frame until the last step: +2 to +3 IoU), camera + radar polar fusion 2411.13311 (fusion on a polar grid at 59 FPS). Gap: these work on dense radar spectra or single sensors; a multi-radar, multi-sweep detection image with a camera on a Cartesian grid is ours.
- Cost (estimate): 5 images of 128 x 132, a 6-layer CNN with 32-64 channels, a few ms; the pull is 12,480 cells x 5 radars, negligible.

## 3a. Option B-lite: smart features + soft rasterisation + two 3x3 convolutions in the polar domain (Mirko, 2026-10-25: "low calculation time, simple")
Hypothesis: the uncertainty need not be decoded by a heavy network if the features carry the physics and the rasterisation already writes an approximate footprint; two small convolutions then only correct it.
1. Per-detection encoder: the 42-value vector of section 1b -> a shared MLP (42 -> 32) = the high-dimensional state of the detection (750 detections for 5 radars and 0.5 s: about 1 M multiply-adds).
2. Soft rasterisation into the sensor's polar image (range x azimuth bins): each detection writes its 32-vector into the 3 x 3 (or 5 x 5) bins around its exact position with analytic weights w = exp(-0.5 ((dr / sigma_r)^2 + (dphi / sigma_phi)^2)), normalised; dr, dphi = the exact offset to the bin centre (continuous, no aliasing), sigma_r and sigma_phi in bins from a small learned table indexed by the quality codes (x_rms, y_rms) and the range. This is the "smart feature design": the prior footprint is built in analytically, in the axis where radar noise is axis-aligned.
   Channels per bin: sum of weights (mass), the weight-averaged 32-vector, the weighted sub-cell offsets, max RCS.
3. Two 3 x 3 convolutions (32 -> 32, ReLU / BN) in the polar domain: they sharpen or widen the prior from the neighbours; total receptive field +-3 bins with a 3 x 3 footprint. Heads (1 x 1): feature c (32) and a likelihood logit p (1); optional log-variance.
4. Fixed polar-to-Cartesian table per radar (gather as in lift PD), V_rad(x, y) = sum over radars c * sigmoid(p).
- Why the convolutions belong in the POLAR domain: there the footprint is axis-aligned and its size in bins is constant over range; in Cartesian 0.5 m cells it rotates with the direction from the sensor and grows with range (1 deg at 50 m = 0.87 m = 1.7 cells, +-3 sigma several cells), so 3 x 3 kernels would not be enough and the orientation would have to be learned from coordinate channels (fallback: CoordConv channels cos / sin of the direction and the range).
- Cost: 16,896 bins (128 x 132) x 2 x (32 x 32 x 9) = 311 M multiply-adds per radar, 1.6 G for five; with a 1 x 1 reduction to 24 channels about 1 G - roughly a millisecond class on the 3060 (estimate, to be measured). Dense, plain PyTorch, no scatter except the small rasterisation (index_add_, about 20 k writes).
- Measured resolution of the nuScenes radar (`temp/radar/radar_resolution.py`): the smallest spacing between detections at the same azimuth is 0.20 m in range (all radars: a range step of about 0.2 m); the azimuth spacing reaches 0.03-0.2 deg at the 1 % quantile and 0.3-0.85 deg at 5 % (so azimuth is a continuous estimate, not a coarse bin); the typical neighbour spacing is much wider (25 % quantile 3-8 deg, 3 m in range). Consequence: polar bins of 0.5 m x 1 deg are coarser than the sensor's own steps, the exact sub-cell offsets are therefore part of the features (nothing is lost by binning), and a footprint of 3 x 3 bins is enough.
- Test that decides it (cheap, synthetic, minutes): targets rendered with sensor noise that grows with range (sigma_r, sigma_phi as in the specification, quality codes as inputs), detections sampled; each variant is trained to output the true target occupancy: (i) Cartesian hard raster + two 3 x 3, (ii) Cartesian hard raster + four 3 x 3, (iii) polar hard raster + two 3 x 3 + gather, (iv) polar soft raster (B-lite) + two 3 x 3 + gather. Prediction: (iv) >= (iii) > (ii) > (i) in IoU beyond 30 m. If (iv) wins by little, the soft rasterisation is dropped; if (i) is as good, polar is not needed.

## 4. Option A: sparse per-detection decoder (Mirko's idea with the refinements) - the control
1. Encoder over the detections of all radars and sweeps (point MLP + local attention among neighbours within a few metres): f_j from (r_j, phi_j, RCS, v_r, static residual, quality codes, states, dt, sensor id / mount).
2. Window per detection: the BEV cells within +-k range bins and +-k azimuth bins of the detection, in its sensor's polar frame (computed per detection from the calibration).
3. Decoder g(f_j, r_cell - r_j, phi_cell - phi_j, r_j, dt) -> a weight logit and a feature; softmax over the window (each detection carries mass 1, or a learned confidence); scatter-add into the radar BEV.
4. Parametric special case (draft of the plan): g = a mixture of two Gaussians in (r, phi) with predicted offsets and sigmas - the uncertainty is explicit and the mixture mean can be checked against the detection.
- Basis: 2101.06742, 2012.10992 (above), RCBEVDet++ 2409.04979 (RCS-scaled scatter, +1.4 NDS), PointPillars 1812.05784 (the hard-bin baseline).
- Cost: about 5 radars x 125 x 6 sweeps = 3,750 detections x 50-170 window cells, a small MLP each - a few ms, a scatter-add (index_add_ in plain PyTorch).

## 5. Comparison
| | B: polar image + pull | A: sparse decoder (push) | parametric A | pillars (R0) |
|---|---|---|---|---|
| symmetric to the camera block | yes (image -> backbone -> c x p -> table) | no (scatter) | no | no |
| keeps the uncertainty | yes: p_rad per polar cell (+ variance) | yes: w per cell | yes, explicit sigmas | no |
| sees neighbouring detections | yes (CNN receptive field) | only through the encoder | only through the encoder | through the BEV trunk |
| exact positions | yes with sub-cell offset channels | yes | yes | no (cell quantised) |
| rig independence | table per radar, shared weights | polar offsets from the calibration | same | no |
| multi-sweep / time | as image channels (dt, Doppler shift before rasterising) | dt into g | dt widens sigma_phi | dt channel |
| run time | dense, gathers, export-friendly | scatter, irregular | scatter | scatter |

## 6. Tests (both options)
- reconstruction: from the radar BEV alone a small decoder recovers the detection cloud - occupancy per cell, positions within the sensor accuracy, Doppler and RCS (linear probe R^2) - against plain rasterisation (R0); B additionally: the polar image itself is reconstructed from its backbone features
- geometry: a synthetic detection at a known (r, phi) of a radar at a random pose lands at the right BEV cell; the same detection seen by two radars at different poses lands at the same cell
- time: a static point measured at several times while the ego moves lands in one cell; a target moving radially at known v is shifted by v * dt
- rig: move a radar in the calibration and transform its detections consistently: the radar BEV is unchanged; drop a radar: only its area changes
- physics of the learned spread: B - the learned footprint of a single detection, in azimuth bins, is about constant over range; A - the mass centre stays near the detection
- usefulness: radar-only auxiliary heads above the null; shuffled radar (from another frame) removes the gain of the fusion

## 7. Open points for Mirko
1. B (symmetric) as the main radar block and A as its control - agreed?
2. Polar resolution: 0.5 m x 1 deg (128 x 132 per radar) or 0.25 m x 0.5 deg (256 x 264)?
3. Detections beyond 60 m (about 40 %): drop them, or keep them in the image (range axis to e.g. 100 m) as context for the backbone even though the BEV ends at 60 m?
4. Sweep window: 0.5 s first, ablation 1 / 5 / 10 sweeps?
