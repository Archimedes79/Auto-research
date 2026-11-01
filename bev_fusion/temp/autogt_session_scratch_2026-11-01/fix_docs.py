p = "autogt/AGENTS.md"
s = open(p, encoding="utf-8").read()
old = """- **f = `share_fusion`**: the share of the footprint (0.125 m samples; samples outside the 60 x 52 m grid count as not observed) where
  the camera sees the object (inside the image at the bottom, middle or top of the box height and not hidden by another box: a pedestrian
  above a low car stays visible) AND the radar fan covers. The states (Mirko, 2026-11-01): care f >= 0.5, partial 0.1 <= f < 0.5, don't
  care f < 0.1; a detection is judged by the same number from `footprint_share(det_corners, fusion_cells(g), grid)` (the ground-plane
  version: a detection has no height). With the OLD masks the helper reproduces the study's per-object table to 0.001."""
new = """- Boxes stand on the ground plane z = 0 of the ego frame (the plane of the map layers and of `camera`; a box on a hill or a bridge is
  placed on it for this purpose). A point inside a box's footprint is never hidden by that box; a box entirely behind the camera hides nothing.
- **f = `share_fusion`**: the share of the footprint (at least 12 x 12 samples, 0.125 m apart for a car; samples outside the 60 x 52 m
  grid count as not observed) where the camera sees the object (inside the image at the bottom, middle or top of the box height and not
  hidden by another box: a pedestrian above a low car stays visible, also when only the head shows) AND the radar fan covers. The states
  (Mirko, 2026-11-01): care f >= 0.5, partial 0.1 <= f < 0.5, don't care f < 0.1. **One rule for ground truth and detections**: a
  detection gets its f from `shares_in(g, corners, height or group=...)` in the frame of the GT dict `g` (everything it needs is in the
  meta: `K`, `T_c2e`, `image_size`, `radar_poses`, the hiding boxes), the stored `share_fusion` of a GT box is the same function.
  `footprint_share(corners, mask, grid)` is the cheaper ground-plane version without a height (the BEV chat's study of
  `bev_fusion/records/footprint_share_study_2026-11-01.md` used it with the old masks; my copy reproduces its per-object table, v5).
- `rig_radar_poses()`: the five radar poses of the nuScenes rig as matrices (mean over the logs, within 0.06 m and 1.5 degrees), for code
  without a calibration at hand, e.g. the model's precompute step: `radar_cells(gx, gy, rig_radar_poses())`."""
assert old in s
s = s.replace(old, new)
old2 = "`T_c2e` (4x4 camera -> ego), `vectors`"
assert old2 in s
s = s.replace(old2, "`T_c2e` (4x4 camera -> ego), `image_size` (width, height of the camera image), `radar_poses` (the five radar -> ego 4x4 matrices),\n`vectors`")
open(p, "w", encoding="utf-8", newline="\n").write(s)

p = "autogt/README.md"
s = open(p, encoding="utf-8").read()
old = """- **Boxes hide with their height** (v7): a box hides the ground behind it with its real volume: the line from the camera to the cell must
  pass through the box's footprint at a height between its bottom and its top (`blocked`, `hidden_cells`). A 0.5 m high box and a camera
  at 1.6 m cast a shadow of about 10 m behind a box that ends at 24 m;"""
assert old in s
s = s.replace(old, """- **Boxes hide with their height** (v7): a box stands on the ground plane z = 0 (the plane of the map layers) and hides the ground behind
  it with its real volume: the line from the camera to the cell must pass through the box's footprint at a height between 0 and its top
  (`blocked`, `hidden_cells`). A 0.5 m high box and a camera at 1.6 m cast a shadow of about 10 m behind a box that ends at 24 m;""")
a = s.index("- **The footprint share f** (D38 b)")
b = s.index("- `num_lidar_pts` / `num_radar_pts` per box")
s = s[:a] + """- **The footprint share f** (D38 b), per box in the meta as `share_fusion` (and `share_camera`, `share_radar`): the share of the footprint
  (at least 12 x 12 samples, 0.125 m apart for a car; samples outside the 60 x 52 m grid count as not observed) where the camera sees the
  object and the radar fan covers. The camera sees a sample when it sees the object above it at the bottom, middle or top of its height
  (inside the image, not hidden by another box), so a pedestrian above a low car stays visible (also when only the head shows). Mirko's
  first states: care f >= 0.5, partial 0.1 <= f < 0.5, don't care f < 0.1. One rule for ground truth and detections:
  `shares_in(g, corners, height)` gives a detection its f in the frame of a GT dict (the meta holds `K`, `T_c2e`, `image_size`,
  `radar_poses` and the hiding boxes); the stored `share_fusion` of a GT box is the same function. `footprint_share(corners, mask, grid)`
  is the cheaper ground-plane version without a height: with the OLD masks (`hidden_cells(soft=False)`) it reproduces the per-object
  table of the BEV chat's study (`bev_fusion/records/footprint_share_study_2026-11-01.md`, 48,724 objects of the val frames).
""" + s[b:]
open(p, "w", encoding="utf-8", newline="\n").write(s)
print("docs rewritten")
