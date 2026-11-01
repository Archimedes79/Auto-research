import numpy as np, sys
g = np.load(sys.argv[1], allow_pickle=True)["gt"].item()["cur"]
for k in ("camera", "radar", "occluded", "undefined", "ignore", "instance"):
    v = g.get(k)
    print(k, None if v is None else (v.shape, float(np.asarray(v).mean())))
m = g.get("meta", {})
print("res", m.get("res"), "cam_file", m.get("cam_file"), "keys", [k for k in g.keys() if k not in ("drivable_area","ped_crossing","walkway","stop_line","carpark_area","lane_divider_dist","lane_divider","road_divider_dist","road_divider","centreline","centreline_dir","vehicle","pedestrian","two_wheeler")][:30])
