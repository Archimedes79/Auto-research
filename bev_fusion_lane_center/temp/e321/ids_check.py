import json, numpy as np
R = "/mnt/c/Auto-research/bev_fusion_lane_center"
ids = np.load(f"{R}/runs/e57_stored/ids_test.npy")
sp = np.asarray(json.load(open(f"{R}/data/splits.json"))["test"], int)
print("ids_test.npy (the row ids of test_clean) equal data/splits.json['test'] in order:", bool((ids.astype(int) == sp).all()), len(ids), len(sp))
