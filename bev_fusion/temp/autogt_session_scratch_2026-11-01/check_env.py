import cv2
import shapely
from nuscenes.map_expansion.map_api import NuScenesMap

print("shapely", shapely.__version__, "cv2", cv2.__version__, "map api ok")
import inspect
print([n for n in dir(NuScenesMap) if n.startswith(("get_map_mask", "discretize", "get_records_in_patch", "extract_polygon", "extract_line"))])
