"""probe: after overfitting one scene, what lane probabilities does the camera-only decoder output on lane cells vs
other cells (is it a threshold / thin-structure problem or is nothing learned)?"""
import sys, torch
sys.argv += ["--quick"]
sys.path.insert(0, "studies/early"); sys.path.insert(0, "training"); sys.path.insert(0, "training/legacy")
import camera_only as T
from camera_decoder import CameraOnlyDecoder
train, _ = torch.load("temp/checks/camera_only_data_150_16_thr0.05_0.5_0.25.pt")
torch.manual_seed(0)
model = CameraOnlyDecoder(grid=T.G).to(T.DEV)
T.train(model, train[:1], 300, 1)
model.eval()
with torch.no_grad():
    p = torch.sigmoid(T.run_model(model, train[:1])["sem"][0]).cpu()
m = T.MASK.bool()
for c, name in enumerate(T.SEM_CLASSES):
    t = train[0][2][c].bool() & m
    o = (~train[0][2][c].bool()) & m
    print(f"{name:12s} p on target cells: mean {p[c][t].mean():.3f} max {p[c][t].max():.3f} | on other cells: mean {p[c][o].mean():.3f} "
          f"p99 {p[c][o].quantile(0.99):.3f} | target cells {int(t.sum())}")
lane = p[0]
dil = torch.nn.functional.max_pool2d(train[0][2][0][None, None], 5, 1, 2)[0, 0].bool() & m
print(f"lanes: mean p within 1 m of a lane {lane[dil].mean():.3f}, elsewhere {lane[m & ~dil].mean():.3f}")
