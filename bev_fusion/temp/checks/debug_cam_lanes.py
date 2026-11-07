"""why are lanes never found by the camera-only decoder? image visibility, target cells in the loss mask, base rate"""
import sys, torch
sys.argv += ["--quick"]
sys.path.insert(0, "tests"); sys.path.insert(0, "training")
import test_camera_only as T
train, test = torch.load("temp/checks/camera_only_data_150_16_thr0.05_0.5_0.25.pt")
m = T.MASK.float()
lane = torch.stack([s[0] for _, _, s, _, _ in train])
veh = torch.stack([s[1] for _, _, s, _, _ in train])
print("mask cells", int(m.sum()), "of", m.numel())
print("lane target cells per scene: all", float(lane.sum((1, 2)).mean()), " inside the mask", float((lane * m).sum((1, 2)).mean()))
print("vehicle target cells per scene inside the mask", float((veh * m).sum((1, 2)).mean()))
print("lane base rate inside the mask", float((lane * m).sum() / (m.sum() * len(train))))
img = train[0][0][0].float() / 255
print("image", tuple(img.shape), "min/max", float(img.min()), float(img.max()))
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(10, 3))
ax[0].imshow(img.permute(1, 2, 0).numpy()); ax[0].set_title("camera frame 0")
ax[1].imshow((lane[0] + 2 * m).flip(1).numpy(), origin="lower"); ax[1].set_title("lane target (1) + loss mask (2)")
fig.savefig("temp/checks/debug_cam_lanes.png", dpi=110)
