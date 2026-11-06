import os
p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "training", "architecture.json")
s = open(p, encoding="utf-8").read()
old_code = '''   "code": [
    "training/extract.py::select_boxes"
   ],
   "note": "select_boxes: the post-processing of model.boxes(); the evaluation adapter still scores with object_head.decode (equal box for box, extract.py self-test), so both stay until the adapter reads select_boxes"'''
assert s.count(old_code) == 1
new_code = '''   "code": [
    "training/extract.py::boxes_from_maps",
    "training/extract.py::select_boxes"
   ],
   "note": "ONE extractor: boxes_from_maps reads only the K peak cells of the head maps (vectorised, float64, the same _params as the dense output) and is what object_head.decode, model.boxes(), the evaluation adapter and the prototype call; select_boxes does the same on a dense output (planner side); extract.py's self-test checks both box by box against the plain per-box loop (6 mutants caught); 1.7 ms instead of 6.5 ms (decode) and 1.5 ms instead of 9.0 ms (model.boxes) per frame at 100 candidates, CPU"'''
s = s.replace(old_code, new_code, 1)
open(p, "w", encoding="utf-8", newline="\n").write(s)
print("ok")
