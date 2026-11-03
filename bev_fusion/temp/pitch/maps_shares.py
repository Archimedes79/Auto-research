"""Like-for-like map shares of the case: our chains and GaussianCaR scored on the SAME 400 frames with the same masks and tolerances (temp/soa_gap/map_ref_<run>_400.json, from pitch.maps_lfl).
usage (bev_fusion folder): python3 temp/pitch/maps_shares.py   -> temp/pitch/maps_lfl.md and temp/pitch/maps_lfl.json
Two readings of the threshold: 'at 0.5' = both systems at 0.5 (as our evaluation); 'best of 3' = each system at its best of {0.3, 0.5, 0.7} per layer (in-sample for both, fair to a model trained with another loss)."""
import json

RUNS = [("t8_ctl_s0", "image features only (control)"), ("t8_bevfeat_s0", "BEVFusion features only"), ("t10_dual_s0", "dual chain, seed 0"), ("t10_dual_s1", "dual chain, seed 1")]
LAY = [("drivable", "drivable_area", "iou@0"), ("walkway", "walkway", "iou@0"), ("carpark", "carpark_area", "iou@0"),
       ("crossing", "ped_crossing", "iou@0"), ("lane divider (0.5 m)", "lane_divider", "iou@0.5"), ("road divider (0.5 m)", "road_divider", "iou@0.5")]
THR = ("0.3", "0.5", "0.7")
sysd = {r: json.load(open(f"temp/soa_gap/map_ref_{r}_400.json")) for r, _ in RUNS}
for r, d in sysd.items():
    assert d["n_frames"] == 400, (r, d["n_frames"])
g = sysd["t10_dual_s0"]["systems"]["gcar"]
for r in sysd:                                              # the GaussianCaR block must be the same in every file (same frames, same logits)
    assert sysd[r]["systems"]["gcar"] == g, f"gcar block of {r} differs"


def val(block, layer, tol, mode):
    """IoU of one layer: at the 0.5 threshold or the best of three thresholds"""
    if mode == "0.5":
        return block[layer]["all"]["0.5"][tol]
    return max(block[layer]["all"][t][tol] for t in THR)


out = {"frames": 400, "modes": {}}
md = ["# Like-for-like map scores on the same 400 frames (GaussianCaR vs our chains)", "",
      "Source: `temp/soa_gap/map_ref_<run>_400.json` (run `pitch.maps_lfl`, GPU, `temp/soa_gap/map_ref_dev.py`), 400 frames of 43 val scenes where GaussianCaR's cached logits exist, the same camera-cell mask and tolerances for both systems, EMA weights, dt 0, cold. "
      "Share = ours / GaussianCaR. One training seed per model except the dual chain (two).", ""]
for mode, title in (("0.5", "both systems at the 0.5 threshold (as our evaluation)"), ("best3", "each system at its best of {0.3, 0.5, 0.7} per layer (in-sample for both)")):
    rows, res = [], {}
    head = "| layer | GaussianCaR | " + " | ".join(f"{n}" for _, n in RUNS) + " |"
    md += [f"## {title}", "", "IoU (share of GaussianCaR in brackets)", "", head, "|" + "---|" * (len(RUNS) + 2)]
    for name, layer, tol in LAY:
        soa = val(g, layer, tol, mode)
        cells, res[name] = [], {"gcar": soa}
        for r, _ in RUNS:
            v = val(sysd[r]["systems"]["ours"], layer, tol, mode)
            res[name][r] = v
            cells.append(f"{v:.3f} ({v / soa:.2f})")
        md.append(f"| {name} | {soa:.3f} | " + " | ".join(cells) + " |")
    out["modes"][mode] = res
    md += [""]
    # dual chain: both seeds
    md += ["Dual chain, the two seeds (share of GaussianCaR): " + "; ".join(
        f"{n} {res[n]['t10_dual_s0'] / res[n]['gcar']:.2f} / {res[n]['t10_dual_s1'] / res[n]['gcar']:.2f}" for n, _, _ in LAY), ""]
open("temp/pitch/maps_lfl.md", "w", encoding="utf-8").write("\n".join(md) + "\n")
json.dump(out, open("temp/pitch/maps_lfl.json", "w"), indent=1)
print("\n".join(md))
