# fiery_ego_warp - FIERY, the ego-warp part only (sections 1, 2, 6, 8 of the template)

| field | value |
|---|---|
| repository | https://github.com/wayveai/fiery (wayveai / fiery), paper 2104.10490 (`papers/2104.10490_fiery.md`) |
| state read | commit `fd03f164935ac172ef6d96117a7a4873abf23919`, 2022-09-21 ("Merge pull request #46 from wayveai/update-readme"), branch master; GitHub API + raw files through the fetch tool's page reader, no clone. `warp_features` and `cumulative_warp_features` were returned as code with line numbers (checked); the rest is digest |
| licence | MIT, "Copyright (c) 2021 Wayve Technologies Limited" (LICENSE file read) |
| extracted by, date | literature scout, 2026-10-25 |

## Port card

| field | value |
|---|---|
| repo | https://github.com/wayveai/fiery |
| commit | fd03f16 2022-09-21 |
| licence | MIT |
| portable | pure-torch (the warp functions); the lift in `fiery.py` uses a custom autograd `VoxelsSumming` but that is not needed |
| reuse | idea-only (the conventions and the test oracle; the function itself assumes a square, centred grid, ours is neither) |
| effort | S |
| port_targets | `bev_fusion/training/temporal_pull.py`, `bev_fusion/training/tests/test_temporal_pull.py` |
| status | read |
| papers | 2104.10490 |

## 1. Map of the code (ego-warp part)

| path | what it does | lines | portable? |
|---|---|---|---|
| `fiery/utils/geometry.py` | `warp_features` (181-222): one 2D rigid warp by `F.affine_grid` + `F.grid_sample`; `cumulative_warp_features` (225-253): warp a sequence to the last frame by accumulating one-step flows; `cumulative_warp_features_reverse` (~252-276): to the first frame; `pose_vec2mat` / `mat2pose_vec` / `euler2mat` (~82-157): 6-DoF vector <-> 4x4 (Rx Ry Rz); `invert_pose_matrix` (~160-178); `calculate_birds_eye_view_parameters` (~39-58) | about 300 | pure torch |
| `fiery/models/fiery.py` | `forward` (~129-200): lift every frame to BEV, `cumulative_warp_features`, optional ego-pose channels, temporal model, decoder | about 300 | needs their encoder and decoder; the wiring is portable |
| `fiery/models/temporal_model.py` | `TemporalModelIdentity` (trims time axis), `TemporalModel` (3D conv stack of `TemporalBlock` + optional `Bottleneck3D` (1,3,3)) | about 60 | pure torch (blocks in `fiery/layers/temporal.py`, not read) |

## 2. Key algorithms (where, how)

**Ego warp, `geometry.py::warp_features(x, flow, mode, spatial_extent)`, line 181** (checked).
- In: `x` [b,c,h,w]; `flow` [b,6] = (tx, ty, tz, rx, ry, rz) of the transform; only tx, ty and the yaw `flow[:,5]` are used; `spatial_extent` = (X_BOUND[1], Y_BOUND[1]) metres = the **half** range of the (symmetric) grid. Out: warped [b,c,h,w].
- Translation is divided by `spatial_extent` ("1.0 = half the image"), then the forward component is negated. The 2x3 `theta` is `[[cos, -sin, ty_n],[sin, cos, -tx_n]]`; the image width axis therefore is the vehicle's y (lateral) and the height axis is x forward, flipped. `theta` maps **output** coordinates to **input** coordinates (backward warp = pull) in normalised [-1,1] coordinates; `affine_grid(theta, size=x.shape, align_corners=False)`, `grid_sample(mode=bilinear|nearest, padding_mode='zeros', align_corners=False)`; out-of-range samples are zero and no validity mask is returned.
- Limitation stated in the code (comment, line 216): rotation preserves distances only if h = w, because the rotation acts in normalised coordinates. Their grid is 200 x 200 and centred on the ego, so it holds; **ours (front-only, 60 m x 52 m, origin at the vehicle edge, 120 x 104 cells) violates both assumptions**.

**Sequence, `cumulative_warp_features(x, flow, ...)`, line 225** (checked). `x` [b,t,c,h,w], `flow` [b,t,6] poses "from t to t+1"; the last frame is untouched, frame `t` is warped by the product `flow[t] @ ... @ flow[-2]` (matrices from `pose_vec2mat`, converted back with `mat2pose_vec` for each warp); every past frame is warped **from its own original** (not re-resampled step by step), so the blur does not accumulate (contrast CRT-Fusion, which re-splats the stacked history at each step).

**Ego-motion channels and temporal block** (`fiery.py::forward`, digest of the page reader, ~lines 129-200).
- Input to the lift `image` [B,S,N,3,H,W], `intrinsics` [B,S,N,3,3], `extrinsics` [B,S,N,4,4], `future_egomotion` [B,S,6]; all cut to the `receptive_field` (default 3) past frames. BEV features per frame `x` [B,S,C,X,Y] -> `cumulative_warp_features(x.clone(), future_egomotion, 'bilinear', spatial_extent)` -> [B,S,C,X,Y] in the present frame.
- If `INPUT_EGOPOSE`: the 6-vector is broadcast to [B,S,6,X,Y], the step-0 entry zeroed and the last step dropped, concatenated on the channel axis -> [B,S,C+6,X,Y]; this goes **into the temporal model**, not into the state.
- `TemporalModel`: permute to [B,C,T,H,W], `receptive_field-1` temporal blocks (3D conv, first one with pyramid pooling `(2,h,w)`), output [B,T-(R-1),C_out,H,W]; for `receptive_field=3` and S=3 the output has one time step, the present state. `TemporalModelIdentity` just trims the same way.

## 6. Port plan

| item | decision |
|---|---|
| licence consequence | MIT: lifting with the copyright notice is allowed; we do not lift the function, we use it as the **sign / convention oracle** and write a metre-based grid |
| what to lift | the idea of `cumulative_warp_features`: warp every past map from its own original with the accumulated transform (no repeated resampling); the zero ego-pose channel at step 0; the unit test "the last frame is unchanged" |
| what to re-write | `temporal_pull.py::ego_pull_grid`: for output cell centres `p_out = (x0 + (i+0.5)dx, y0 + (j+0.5)dy)` compute `p_src = T_{s<-c} p_out` (SE(2) from the full 4x4, yaw and translation in metres), normalise per axis with `u = 2 (p_src - origin) / (N d) - 1` (`align_corners=False`), no square or centred-grid assumption; return grid and a validity mask |
| dependency gaps | none (`F.grid_sample`) |
| effort, risk | S; risk is the axis convention (their width = lateral y, height = forward x flipped): fix our (row, col) <-> (x, y) convention once in a test with an asymmetric blob |

## 8. Snippets (MIT, at most 15 lines, `fiery/utils/geometry.py` at fd03f16)

Lines 205-220, build of `theta` and the sampling (copyright (c) 2021 Wayve Technologies Limited, MIT):
```python
cos_theta = torch.cos(angle)
sin_theta = torch.sin(angle)
transformation = torch.stack([cos_theta, -sin_theta, translation[:, 1],
                              sin_theta, cos_theta, translation[:, 0]], dim=-1).view(b, 2, 3)
grid = torch.nn.functional.affine_grid(transformation, size=x.shape, align_corners=False)
warped_x = torch.nn.functional.grid_sample(x, grid.float(), mode=mode, padding_mode='zeros', align_corners=False)
```
