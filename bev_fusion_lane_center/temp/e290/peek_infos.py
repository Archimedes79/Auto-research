import sys
sys.path.insert(0, "../../tools/viz")
import numpy as np
import qgraph_view as QV

inf = QV._infos()
tok = QV.val_scenes()[11][1][3]
i = inf[tok]
print(list(i.keys()))
ci = i["cam_infos"] if "cam_infos" in i else None
print(type(ci), list(ci.keys())[:7] if ci else None)
c = ci["CAM_FRONT"]
print(list(c.keys()))
print(np.round(np.asarray(c["ego_pose"]["translation"]), 2), c["ego_pose"]["rotation"], c["ego_pose"].get("timestamp"))
print(i.get("timestamp"))
