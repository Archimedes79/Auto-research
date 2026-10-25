"""E220 summary: route success of the best chain (map builder B + gate, exact map, fork pruning 0.3) with the original and the failure-trained route heads, map error 0 / 2 m, clean and failure half. Usage: p18_e220_table.py"""
import json

D = "/mnt/c/ML-Experiment/bev_fusion_lane_center/runs/e17/e18_route_best"
for tg in ("dark_gateB", "blur_gateB", "cam_front3_off_gateB"):
    j = json.load(open(f"{D}/route_{tg}.json"))
    print("==", tg)
    for sg in ("0", "2"):
        row = []
        for rt in ("p16_f2hr_route_s0", "p16_f2hr_route_s1", "e18_joint_c0", "e18_joint_c1"):
            cl, fl = j[f"sigma{sg}_{rt}_cl"]["route_success"], j[f"sigma{sg}_{rt}_fail"]["route_success"]
            row.append(f"{rt[-9:]}: clean {cl:.3f} fail {fl:.3f} (n {j[f'sigma{sg}_{rt}_fail']['n']})")
        print(f"  map error {sg} m | " + " | ".join(row))
