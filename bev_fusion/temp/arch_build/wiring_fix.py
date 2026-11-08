"""one-off: the drawing's wiring after the design review (run from bev_fusion/): backbone -> time mover -> heads (the shared TimeMover
moves S to dt before both heads), the camera visibility from the precompute step into the temporal block, the planned edges dashed,
the ego steps named as built from the GT meta in train.py today (a layer-1 leak), select boxes not yet the metrics' path"""
import json

p = "training/architecture.json"
s = json.load(open(p, encoding="utf-8"))
E = s["edges"]
E[:] = [e for e in E if not (e["from"] == "backbone" and e["to"] in ("maps_head", "obj_head"))]
E[:] = [e for e in E if not (e["from"] == "cam_enc" and e["to"] == "temporal" and e.get("label") == "visibility")]
E[:] = [e for e in E if not (e["from"] == "motion" and e["to"] == "out_iface")]
for e in E:
    if e["from"] == "backbone" and e["to"] == "motion":
        e["label"] = "S"
    if e["from"] == "vis" and e["to"] in ("loss", "consumers"):
        e["dashed"] = True
        e["label"] = e["to"] == "loss" and "loss masks (D36, next)" or "scoring masks (D37, next)"
    if e["from"] == "vis" and e["to"] == "iface":
        e["label"] = "memory write masks"
E += [{"from": "motion", "to": "maps_head", "label": "S at dt"}, {"from": "motion", "to": "obj_head"},
      {"from": "motion", "to": "out_iface", "label": "v_hat"}]
N = {n["id"]: n for n in s["nodes"]}
N["motion"]["note"] = "the shared TimeMover: moves S to each target time dt (velocity head + ego splat) before both heads, gives v_hat at dt 0"
N["pre_ego"]["note"] = ("the ego step from every past frame to now (dataset, from the poses), calibration and crop; TODAY the true gap to the "
                        "target time and the memory step between calls are built in train.py from the GT meta (batch_ego, clip_step): a "
                        "layer-1 leak, the network cannot run without GT until the dataset makes them from the poses")
N["decode"]["note"] = ("select_boxes: the post-processing of model.boxes(); the evaluation adapter still scores with object_head.decode "
                       "(equal box for box, extract.py self-test), so both stay until the adapter reads select_boxes")
json.dump(s, open(p, "w", encoding="utf-8", newline="\n"), indent=1, ensure_ascii=False)
print(len(s["edges"]), "arrows")
