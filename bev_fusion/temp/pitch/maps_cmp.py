"""GPU path against the stored CPU path of the like-for-like map scoring: the same model (t1_r4_soa, 400 frames) scored twice.
usage: python3 temp/pitch/maps_cmp.py <cpu.json> <gpu.json>; exit 1 when a headline number of 'ours' differs by more than 0.01 or the 'gcar' block is not identical"""
import json
import sys

LAY = [("drivable", "drivable_area", "iou@0"), ("walkway", "walkway", "iou@0"), ("carpark", "carpark_area", "iou@0"),
       ("crossing", "ped_crossing", "iou@0"), ("lane divider @0.5", "lane_divider", "iou@0.5"), ("road divider @0.5", "road_divider", "iou@0.5")]


def walk(a, b, path=()):
    """yield (path, a, b) for every number the two nested dicts share"""
    for k, v in a.items():
        if isinstance(v, dict):
            yield from walk(v, b[k], path + (k,))
        else:
            yield path + (k,), v, b[k]


cpu, gpu = (json.load(open(p))["systems"] for p in sys.argv[1:3])
worst = {s: max(abs(x - y) for _, x, y in walk(cpu[s], gpu[s])) for s in ("ours", "gcar")}
print("max abs difference over every number (layer / distance bin / threshold / tolerance):", {s: round(w, 5) for s, w in worst.items()})
head = []
for n, l, t in LAY:
    c, g = cpu["ours"][l]["all"]["0.5"][t], gpu["ours"][l]["all"]["0.5"][t]
    head.append(abs(c - g))
    print(f"  {n:18s} cpu {c:.4f}  gpu {g:.4f}  diff {g - c:+.4f}")
ok = max(head) <= 0.01 and worst["gcar"] == 0
print("headline max diff", round(max(head), 5), "->", "GPU path reproduces the CPU numbers" if ok else "MISMATCH")
sys.exit(0 if ok else 1)
