"""one-off: the built design becomes the one drawing of bev_fusion/training: architecture_target.json -> architecture.json with every box
pointing at the code that now implements it (run from bev_fusion/)"""
import json

s = json.load(open("training/architecture_target.json", encoding="utf-8"))
s["title"] = "bev_fusion: camera + radar BEV system in five layers"
s["subtitle"] = ("Built 2026-11-01 (Mirko's layered design). Left of the input interface nothing learns; between the two interfaces every "
                 "slot is a swappable block chosen in ModelConfig (one builder table per slot, tests/test_slots.py) with no rule step; "
                 "right of the output interface only the selection of boxes and line cells for the metrics, and later the planner.")
N = {n["id"]: n for n in s["nodes"]}


def put(i, **kw):
    N[i].update(kw)


put("vis", code=["training/precompute.py::Precompute", "training/precompute.py::radar_visibility", "training/lift_pd.py::camera_visibility"],
    note="the precompute step (precompute.py): camera map from calibration + grid (cached per calibration, the same function live), radar "
         "map from the detections; read by the network (memory write masks of both branches), the training and the evaluation (D36: the "
         "loss and scoring masks move onto it); the GT Discussion chat's autogt.coverage_for replaces the two definitions in this one place")
put("iface", code=["training/interface.py::check_inputs", "training/nuscenes_dataset.py::collate", "training/feature_cache.py::collate_cached"],
    note="BEVInputs = the batch dict with its contract in interface.INPUTS (names, shapes, units, frames) and check_inputs(); the forward "
         "arguments dts, state, step, ego in interface.CALL; the per-sample flip flag travels with it (the mirror is a BEV operation after the camera slot)")
put("out_iface", code=["training/interface.py::check_outputs", "training/model.py::BEVFusionModel.forward"],
    note="BEVOutputs = per dt the maps, the object maps, the dense boxes and the soft line map (interface.OUTPUTS_PER_DT / OPTIONAL_PER_DT, "
         "check_outputs), plus v_hat, the features S and the temporal state")
put("trainer", label="Trainer + slot tables\ncheck_config", code=["training/train.py::train", "training/model.py::check_config",
                                                                  "training/feature_cache.py::CachedBEVFusionModel"],
    note="ModelConfig picks one kind per slot from the builder tables of model.py (LIFTS, COLLAPSES, TEMPORALS, RADARS, FUSIONS, "
         "BACKBONES); check_config refuses every unknown kind and unfitting combination in one place")
put("temporal", note=N["temporal"]["note"] + "; finding (tests/test_slots.py): with no past radar frames the window part and the velocity "
                     "head of the radar block get no gradient, so the radar memory needs its own static gate (the measured Doppler) or past radar BEVs")
s["steps"][1]["label"] = "1 Precompute (own step)"
for g in s["groups"]:
    if g["label"].startswith("3 network"):
        g["label"] = "3 network: one slot per block, the kind chosen in ModelConfig, a contract test per kind"
json.dump(s, open("training/architecture.json", "w", encoding="utf-8", newline="\n"), indent=1, ensure_ascii=False)
print(len(s["nodes"]), "boxes")
