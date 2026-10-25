# Fixes for the four findings of the synthetic gates (scout of 2026-10-25)

Fact tags: `checked` = read on a page or raw code file this session (all web reads went through a page summariser, quotes are paraphrases), `digest` = from a search summary or abstract only, `memory` = from my memory, `derived` = my arithmetic from stated assumptions. Notes of the new papers are in `papers/` (ids below), rows in `temp/scout_tables/papers_new_fixes.csv`. Nothing was downloaded or cloned.

## 1. Lift by point sampling misses thin surfaces and far range

### 1a. How the pull lifts choose heights (checked unless marked)
| method | heights | sampling | what it means for a thin road surface |
|---|---|---|---|
| Simple-BEV 2206.07959 (MIT) | 8 voxel cells over Y in [-5, 5] m = 1.25 m per cell, 0.5 m on the ground (X, Z in [-50, 50] m, 200 x 8 x 200) | one bilinear `grid_sample` per camera at each voxel CENTRE, zero padding, validity mask, camera mean | one point per 1.25 m thick cell; the road is caught only because a cell centre happens to lie within about 0.13 m of it (derived: ground 1.5 m below the camera); no depth, every voxel on a ray gets the same value |
| BEVFormer 2203.17270 (Apache-2.0 memory) | 4 pillar points per cell, `linspace(0.5, Z-0.5, 4)/Z`, z range [-5, 3] m (config) = z of -4.5, -2.17, 0.17, 2.5 m (derived) in the lidar frame | projected with lidar2img, then DEFORMABLE attention with 8 learned offsets around them | the ground (about -1.8 m in the lidar frame, memory) is 0.4 m from the nearest reference point; the learned offsets, not the reference points, close the gap |
| Fast-BEV 2301.12511 | 200 x 200 x 4 or x 6 voxels (z extent not on the page) | pre-computed index table, nearest pixel, first camera that sees the voxel, uniform along the ray | nearest-index, no interpolation; thin structures rely on the voxel count along z |
| DualBEV 2403.05402 (Apache-2.0), `view_transformer_dual.py` | 13 fixed heights in [-4.5, 2.5] m of the lidar frame: 0.5 m spacing from -2.5 to 2.5 plus 1 m steps below, so z = 0 IS included, and the ground (about -1.8 m, memory) lies within 0.25 m of a sample (derived) | projected point -> ROUNDED to the nearest feature pixel AND nearest depth bin (the code, not the paper's 'trilinear'); the 13 heights and all cameras are summed into one cell (Z = 1) | dense 0.5 m spacing is the reason the road is never missed; the sum over 13 samples is a crude integral over the pillar |
| OFT 1811.08188 | 4 m of 0.5 m voxels below the camera | average of the image features in the voxel's projected RECTANGLE via an integral image, height collapsed by learned weights | the only pull lift that integrates over an extent (in the image plane); anti-aliasing at far range |
| BEV-SAN 2212.01231 (new) | global slices [-6,4], [-5,3], [-4,2] m and local slices chosen from a LiDAR height histogram (peaks near -2 and 0) | one LSS lift per slice | NDS +0.007 for lidar-guided over evenly spread slices, +0.036 over a single plane; the histogram peak IS the ground |

Reading: no published pull lift samples a thin slab with one point; they either use dense sample spacing (DualBEV 0.5 m), thick cells where the ground happens to fall at a centre (Simple-BEV), learned offsets (BEVFormer), or an image-plane average (OFT). Our finding (road at z = 0 between the slice centres) is the expected failure of a few-point sample, and our `oversample=4` (mean of 4 sub-heights per slice) is the same thing as DualBEV's dense spacing in a Z-as-channels layout.

### 1b. Depth targets (checked)
- BEVDepth 2206.10092 (MIT), `base_exp.py`: lidar depth reshaped in 16 x 16 blocks, zeros replaced by 1e5, MIN taken, bin index `(d - (dmin - step))/step` with d_bound [2, 58, 0.5] = 112 bins, `F.one_hot` and drop of the 'no label' column, BCE on foreground cells only (a cell with at least one positive bin), divided by the number of such cells, weight 3.0.
- DualBEV: same one-hot minimum, but for cells with semantic foreground the hard label is replaced by the normalised histogram of the block's depths (soft label), foreground weight 0.9 vs background 0.1, loss weight 3.0 (code note `literature/code/dualbev.md`, checked there).
- CaDDN 2103.01100: LID bins, focal loss with foreground weighting; whole distribution lifted (existing note); whether its ground truth is one-hot of a dense completed depth is from memory.
- DualPathOcc 2609.06370 (new): Occ3D-nuScenes val mIoU one-hot LiDAR depth 35.96 < no depth loss 36.92 ~ Gaussian-softened target (sigma 1.5 bins) 36.94 [checked]. One-hot of the first surface hurts in a densely supervised BEV task.
- GeoBEV 2409.01816: 'In-Box' label of the object volume instead of the surface [digest, abstract only].
- Why the minimum fails at far range (derived, assumptions f = 633 px at half-resolution, camera height 1.5 m): a stride-8 row group of road pixels spans dz = 8 z^2/(f h) in depth = 0.8 m at 10 m, 3.4 m at 20 m, 21 m at 50 m. The min-pool target puts all mass at the nearest depth of the group, so the farther part of the road in that group gets probability about 0 in the pull, which is exactly the lane recall 0.89 / 0.21 / 0.08 with a one-hot minimum depth.

### 1c. Integrating the depth distribution over the voxel's extent
- Exists as: (i) CaDDN, trilinear over 2 depth bins and 2x2 pixels at the voxel centre (a fixed small footprint; checked in the existing note); (ii) FB-BEV, linear interpolation between the two adjacent bins (existing note); (iii) GaussianLSS 2504.01957, a Gaussian whose extent comes from the depth variance, rendered into the BEV cell (existing note); (iv) Mip-NeRF 2103.13415 / Zip-NeRF (existing notes) as the maths of integrating a Gaussian over a cell. No paper integrates a depth DISTRIBUTION over the voxel's depth extent in a pull lift; our PD-G (plan 2.4: exact mixture mass Phi - Phi over [1/(z+d/2), 1/(z-d/2)]) would be new.

### Recommendation 1
1. Keep PD with `oversample=4`; put a slice centre (or a dense 0.25-0.5 m sample spacing as DualBEV) on z_car = 0 and say why in the table: the oracle test already shows the road recovered (road 1.00 / 1.00). Optionally take the slice heights from a height histogram of the vehicle-frame GT (BEV-SAN's recipe, free for us from the GT).
2. Depth target for the head fine-tune: histogram of the whole 8 x 8 group (DualBEV's soft label for ALL cells, not only foreground) smoothed along D with a Gaussian (DualPathOcc, sigma 1-2 bins) as the cheap arm; compare with raw histogram and with the 1/z mixture of plan 2.4 by the oracle lane recall at 40-60 m. Do not use the minimum for road or lane cells.
3. For road and lane pixels the depth within a group is analytic under the flat-ground hypothesis (depth = f h / (v - v0), dz/dv = z^2/(f h)): an additional analytic ground-depth term (BEVHeight 2303.08498 / MonoGround 2206.07372 in our table) would cover the far lanes without any depth head; keep it as an arm, it is also the null for a learned depth.
4. Do NOT port DualBEV's nearest rounding (no gradient to the heights); the trilinear form in our port card is the right one.

## 2. Chained recurrent states blur and moving objects lag

### What exists (checked)
| method | how the past enters | moving objects | measured |
|---|---|---|---|
| BEVDet4D 2203.17054 | previous BEV ego-warped and concatenated (window of 1) | implicit in the feature difference | NDS 39.2 to 44.0, mAVE 0.909 to 0.479; unaligned worse than none (37.6) (existing note) |
| BEVFormer 2203.17270 | recurrent BEV, deformable temporal attention with offsets from [query, aligned history] | offsets absorb motion | ego alignment alone only +0.007 NDS; mAVE 0.802 to 0.394 (existing note) |
| SOLOFusion 2210.02443 | parallel stack of up to 16 low-res BEV frames | ego only | NDS 0 / 1 / 16 / 41 past frames 0.347 / 0.423 / 0.474 / 0.467: one frame already gives 60 % of the gain, saturates at 16 |
| VideoBEV 2303.05970 | recurrent, H_i = [warp(H_i-1, ego); B_i] * V, time-gap embedding | ego only (page says no object alignment) | val R50: recurrent 8 frames 0.535 NDS / 0.286 mAVE vs SOLOFusion parallel 17 frames 0.534 / 0.252 [checked]; no equal-frame comparison |
| OnlineBEV 2507.08644 (new) | recurrent single memory + motion-guided deformable alignment + consistency loss | learned offsets from the query difference | +1.0 NDS and mAVE 0.296 to 0.287 from the alignment, +0.5 NDS from the loss; 44.4 mAP / 54.5 NDS vs SOLOFusion 42.7 / 53.4 |
| StreamPETR 2303.11926 (new, Apache-2.0) | list of 4 x 256 object queries, centres moved by exact ego transform | MLN conditioning on [ego, velocity, dt] | explicit shifting alone did not help (NDS 0.481); implicit encoding +2.0 mAP; memory 1 frame gives most (NDS 0.372 to 0.501), saturates at 2; 27.1 vs 27.7 FPS |
| Sparse4D v2 2305.14018 (new, MIT) | list of instances; only the ANCHOR is projected (position + v dt, then ego), feature copied | explicit velocity move of the coordinates | cost O(1): 19.4 FPS and 432 MB for any T vs v1 12.6 / 9.0 / 6.1 FPS for T = 3 / 5 / 9; no matched NDS |
| CRT-Fusion 2411.03013 | per-cell velocity shift (push) | explicit | no isolated ablation; code line likely a no-op (existing note) |

Honest gap: NO paper measures the lag or blur of moving objects for recurrent vs window at equal frames. The closest are VideoBEV (different frame counts) and StreamPETR's statement that dense BEV memories misalign moving objects (argument, no matched experiment).

### Recommendation 2
- What we found (chain coordinates, splat each original once) is what Sparse4D v2 and StreamPETR do at object level: coordinates moved by exact transforms, content copied. Keep `chain_positions` + `splat_window`; this is the right design, not an outlier.
- Window length: take K = 2-4, not more. Evidence: BEVDet4D/SOLOFusion (1 past frame = about 60 % of the gain), StreamPETR (saturation at 2 frames), MemFusionMap (4).
- Keep the recurrent state for the static map only (ego-only chains well; VideoBEV-type decay conv). Moving content is rebuilt from the window every frame.
- Add OnlineBEV's label-free consistency loss as the training signal for the learned correction: the same head on the current BEV and on the aligned history must agree (stop-gradient on the current side). This replaces the 1 x 1 position-offset head that failed on chained maps: let the correction see spatial context (3 x 3 conv or deformable offsets from the difference of current and warped history), not a per-cell MLP.
- For the object head (plan 8.3) propagate object centres by Sparse4D's rule (position + v dt, ego rotation), features copied, and condition on [ego, v, dt] as StreamPETR's MLN does.

## 3. Flow / velocity field between two BEV frames

### What exists
| family | method | how | numbers (checked) |
|---|---|---|---|
| supervised, camera BEV | FIERY 2104.10490, PowerBEV 2306.10761 (new, MIT), Cam4DOcc 2311.17663 [digest] | backward centripetal flow: foreground pixel at t -> its instance centre at t-1, smooth L1; pulled by `grid_sample`; box GT | PowerBEV nuScenes IoU / VPQ 62.5 / 55.5 (short), 39.3 / 33.8 (long) vs FIERY 59.4 / 50.2, 36.7 / 29.9; warping association 14 ms vs 82 ms; no no-flow ablation |
| supervised, lidar BEV | MotionNet 2003.06754 (coarse-to-fine pyramid, pseudo-1D temporal convs); FastFlow3D 2103.01306 [digest] | direct 2D displacement per pillar | MotionNet error static 0.026 m, slow 0.257, fast 1.074 (existing note) |
| label-free, lidar BEV | SPML 2104.08683 (new, CC BY-NC-SA) | warped-points chamfer to the next sweep with a probabilistic static mask, camera-flow L1 (0.01), smoothness | error static / slow / fast 0.162 / 0.697 / 3.550 vs zero motion 0.000 / 0.611 / 8.652 (ContrastMotion table) vs supervised MotionNet 0.024 / 0.247 / 1.011; fine-tuned 0.0245 / 0.229 / 0.778 |
| label-free, lidar BEV | ContrastMotion 2304.12589 (new, no code) | association by softmax over pillar-feature distance in a 32-patch, contrastive training on an augmented copy with a known transform | 0.083 / 0.452 / 3.527 (same table) |
| label-free, camera | SelfOccFlow 2602.23894 (new, preprint, no code) | cosine similarity of BEV features t vs t+-1 in a 35 x 35 window -> arg-max displacement as pseudo-label, forward-backward weight | mAVE 1.31 vs 1.61 for 3D-supervised OccNet on nuScenes; without the cue no flow is learned |
| label-free, camera | Let Occ Flow 2407.07587 (new) | rendered flow + pretrained 2D optical flow minus static flow | foreground EPE 9.66 to 7.51 with the flow cue on KITTI-MOT; 253 M parameters |
| label-free, image | RAFT 2003.12039, SMURF 2105.07014 [digest] | all-pairs correlation volume, iterative refinement; self-supervision by warping, multi-frame self-teaching | SMURF 36-40 % better than UFlow [digest]; no BEV use found |
| alignment as latent flow | OnlineBEV | consistency of the heads on current vs aligned history | see 2 |

No BEV-specific RAFT/correlation-volume flow network was found in this scout; the label-free camera/lidar methods use either a local feature correlation (ContrastMotion, SelfOccFlow) or a point/chamfer match (SPML).

### Recommendation 3 (for the gate row 'temporal: velocity / flow from data')
1. Supervised arm (a): FIERY/PowerBEV style, backward flow to the t-1 position as the target and a masked smooth L1 on object cells; v_hat = flow / dt. Already planned from CRT's GT recipe; the pull form needs no hole filling.
2. Self-supervised arm (b) = SPML structure on rasters: loss = |splat(past, v_hat) - observed|^2 weighted by mass (only where both mass maps > 0: our 'unknown' cells are masked out), with SPML's static weighting exp(-alpha max(|v| - tau, 0)) so ego-compensated static cells carry the loss and moving cells are not smeared, plus a smoothness term. Expect it to identify motion only where the BEV has structure (edges, aperture problem, derived): do not expect an inside-object velocity from a smooth interior; rigidity (flow constant inside the object) or SPML's fine-tune with a few labels covers it.
3. Network: one-level local correlation of the two feature maps in a +-2 m window (9 x 9 at 0.5 m covers 15 m/s at 12 Hz, derived), softmax-argmax (ContrastMotion/SelfOccFlow), then a small conv refinement; add coarse-to-fine only if the fast bucket stays bad.
4. Free self-supervision on the synthetic world: random known rigid augmentation of one frame as the pair (ContrastMotion), plus the exact ego-induced flow subtracted before learning (Let Occ Flow's decomposition).
5. Expected level: label-free reaches about 40 % of the zero-motion error in the fast bucket (3.5 vs 8.65), not the supervised 1.0; plan the label-free arm as an initialisation, as SPML shows (fine-tuned 0.78 vs supervised 0.95).

## 4. Radar velocity: one return gives the radial Doppler only

### What exists
| approach | method | how | numbers |
|---|---|---|---|
| back-projection | RadarNet 2007.14366 (existing) | v_r / cos(phi) along the detected motion direction | AVE -14 % (existing note) |
| BEV learned | CR3DT 2403.15313 (new), RC-BEVFusion 2305.15883 (existing) | 5-sweep accumulation, pillar average with velocity components, concatenated with camera BEV | mAVE 0.47 vs 0.86 (CR3DT, nuScenes val); 0.918 to 0.421 (RC-BEVFusion) |
| radar scene flow, label-free | RaFlow 2203.01137 (new) | radial-displacement loss: flow projected on the line of sight = Doppler x dt, soft Chamfer, smoothness; static set from the radial residual (<= 15 %) + Kabsch ego motion | in-house 4D radar: EPE 0.570 m, radial error 0.240 m, motion seg accuracy 81.9 % |
| radar scene flow, cross-modal supervision | CMFlow 2303.00462 (note upgraded; MIT code https://github.com/Toytiny/CMFlow) | adds odometer, LiDAR-tracker pseudo flow, RAFT camera flow | View-of-Delft EPE 0.226 (RaFlow) to 0.141; odometer alone gives 0.161 |
| radar + camera flow, closed form | Long et al. 2108.10637 (new, no code) | tangential part from the camera optical flow at the associated pixel by a 3 x 3 solve; learned association (U-Net, 2.6 ms) | nuScenes point-wise full-velocity error 0.433 m/s (raw projection 0.577, radial-only 1.599), tangential 0.322 vs 1.536; objects 0.451 vs CenterFusion 0.826; flow 0.447 s per frame (RAFT) |
| radar-only, several returns | Kellner et al. (FUSION 2013, ICRA 2014, IV 2014: 'Instantaneous full-motion estimation of arbitrary objects using dual Doppler radar') [digest: titles and abstract of the IV paper only] | a rigid object seen by several returns / two radars: v_r,i = v . u_i with u_i the unit line of sight; two sensor positions or enough azimuth spread give the 2D velocity (and yaw rate) in one cycle without a motion model | no numbers read; conditioning grows as 1 / sin(azimuth spread): a 4 m car at 40 m spans about 0.1 rad, so the tangential error is about 14 x the Doppler noise (derived) |
| multi-sweep with Doppler | DoppDrive 2508.12330 (existing) | Doppler-driven temporal aggregation (radial shift of old points) | see existing note |

### Recommendation 4
- Physics first: radial Doppler d_i = u_i . v is exact once ego motion is removed. Per cell the velocity head should output v (2 numbers) with the RADIAL component constrained: v_hat = v_free + u (d - u . v_free) when a Doppler measurement exists in the cell (projection onto the Doppler constraint, weight by sigma_d vs sigma_flow). Spec test (synthetic, known answer): given d = u . v exactly, the radial error of v_hat is 0 to float precision, and the tangential error equals that of v_free.
- Tangential from a second observation: the camera BEV flow of problem 3 (Long et al.'s closed form does the same point-wise with an image flow: 0.322 m/s tangential on nuScenes) or the radar's own previous sweep with association (RaFlow's Chamfer term; tangential EPE stays the weak point at 0.57 m).
- Rigid-object fit as a cheap radar-only baseline: least squares of d_i = u_i . v over the returns inside a predicted object footprint (2 unknowns, needs azimuth spread; the Kellner family), valid only where >= 2 returns with spread exist; use it as a prior on object cells, not as the main route.
- Labels: ego pose gives most of the radar-flow gain (CMFlow: 0.228 to 0.161), so ego-removed Doppler needs no extra labels; keep the nuScenes quirk in mind (the devkit does not rotate velocity vectors when merging sweeps, existing note NUSC_RADAR).
- Baseline arm to beat: CR3DT-style sweep accumulation + pillar average with compensated velocity components (mAVE 0.47 vs 0.86 camera-only).

## What the next scout should look at
- A measured recurrent-vs-window comparison for moving objects in a dense grid (none found); dense occupancy-flow codebases (Cam4DOcc, FlowOcc) for the flow-head code.
- BEV-level radar+camera full-velocity (a per-cell version of Long et al.) and any code for the Kellner rigid-body fit.
- Code of the label-free flows (SPML repo is non-commercial: re-implement the loss; ContrastMotion and SelfOccFlow have no code).
