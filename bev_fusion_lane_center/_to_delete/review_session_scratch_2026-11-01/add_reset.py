"""throw-away: t11 configs -> arms A' / B' (fresh vel branch, rule C1 of t11_vel_cause)"""
import json
about = {"t11_vel_ctl": "arm A' of experiments/t11_vel_hurdle: warm start from t10_dual_s0 with a FRESH vel branch (init_reset, rule C1 of "
                        "t11_vel_cause: the vel ReLU layer is dead in that checkpoint) + today's L1 on all centres; 1000 steps, seed 0",
         "t11_vel_hurdle": "arm B' of experiments/t11_vel_hurdle: warm start from t10_dual_s0 with a FRESH vel branch (init_reset, rule C1 of "
                           "t11_vel_cause) + the hurdle (model.object_moving: moving / static logit, velocity L1 on moving centres only); "
                           "1000 steps, seed 0"}
for name, text in about.items():
    p = f"training/configs/{name}.json"
    c = json.load(open(p))
    c = {"_about": text, **{k: v for k, v in c.items() if k != "_about"}, "init_reset": ["obj_head.head.vel."]}
    with open(p, "w") as f:
        json.dump(c, f, indent=1)
        f.write("\n")
