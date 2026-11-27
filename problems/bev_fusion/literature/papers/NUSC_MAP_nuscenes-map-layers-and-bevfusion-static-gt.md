# NUSC_MAP nuScenes map expansion layers and how camera-radar BEV networks cut their static GT from them

| | |
|---|---|
| source | https://raw.githubusercontent.com/nutonomy/nuscenes-devkit/master/python-sdk/nuscenes/map_expansion/map_api.py (layer lists), https://github.com/nutonomy/nuscenes-devkit/blob/master/docs/schema_nuscenes.md (map table), https://raw.githubusercontent.com/mit-han-lab/bevfusion/main/mmdet3d/datasets/pipelines/loading.py (LoadBEVSegmentation) |
| fact status | checked (source pages read through a page summariser, 2026-10-26); the map_api page was cut at 100 k of 101 k characters, the layer lists were inside the read part |
| groups | A1: 2, A8: 2, F4: 2 |
| code / licence | nuscenes-devkit Apache-2.0 (memory); bevfusion Apache-2.0 (memory); map data under nuScenes terms (non-commercial) |
| verdict | use (data facts: what the map can and cannot provide as a radar target) |

**What it is**: The vector map has polygon layers `drivable_area, road_segment, road_block, lane, ped_crossing, walkway, stop_line, carpark_area` and line layers `road_divider, lane_divider, traffic_light` (devkit `non_geometric_polygon_layers` / `non_geometric_line_layers`, checked). No layer contains curb, guard rail, barrier, fence or wall (checked: none of those words occurs in the layer lists). The schema `map` table only mentions a semantic prior for drivable surface and sidewalk (the raster maps).

**How the networks use it** (checked): BEVFusion (MIT) `LoadBEVSegmentation` rasterises with `NuScenesMap.get_map_mask()`: `drivable_area*` = `road_segment` + `lane`, `divider` = `road_divider` + `lane_divider`, any other class name = the layer of the same name (`ped_crossing`, `walkway`, `stop_line`, `carpark_area`); the grid comes from the config (xbound, ybound); vehicles are not part of this loader (they come from boxes). BEVCar (2403.11761) evaluates seven map classes (drivable area, carpark area, ped crossing, walkway, stop line, road divider, lane divider) with an alpha-balanced focal loss; RCBEVDet++ (2409.04979), GaussianCaR (2602.08784) use drivable area and lane (divider) without naming the label source on the page. None adds a radar-specific map target.

**Consequence for a radar stationary-structure target**: the only physical-edge information the map holds is the border of `drivable_area` and the border between `drivable_area` and `walkway` / `ped_crossing` (a kerb, inference). Guard rails, walls, fences, bushes and poles, which dominate static radar returns (qualitative support only: digest from search snippets, no per-class share found), are not in it. A map-edge band is therefore a road-boundary prior, not a measurement of what the radar sees; its adequacy must be measured (precision / recall against Doppler-static returns) before it is used as a loss.

**Cost and fit**: free, already in `training/gt_bev.py`.

**Doubts**: our own map registration offset (0.5-1 m in places, plan 9.5) is as large as the 1 m band proposed in 8.4.
