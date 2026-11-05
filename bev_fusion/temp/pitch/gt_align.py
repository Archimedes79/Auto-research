import sys
sys.path.insert(0, ".")
import evaluation.bev_gtcheck as G
samples = G.pick(24, "15efbde2d1", 0)
iou, dev = G.alignment(samples)
for (layer, dt, tag), v in sorted(iou.items()):
    print(f"{layer:14s} dt{dt:+.1f} {tag:9s} IoU {v:.3f}")
print("pose vs steps displacement, mean abs difference (m):", round(dev, 3))
