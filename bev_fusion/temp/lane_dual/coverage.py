"""scratch: does data/cache/img_front hold the CAM_FRONT key-frame image of every sample of the large split (train and val datasets of the t20 config)?"""
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path[:0] = [os.path.join(ROOT, "training"), ROOT, os.path.join(ROOT, ".."), os.path.join(ROOT, "..", "autoresearch")]
os.chdir(ROOT)
import train as T
from lane_branch import image_path
cfg = T.read_config("temp/compose/t20_check.preview.json")
cfg.workers = 0
dss = T.make_datasets(cfg)
for name, ds in zip(("train", "val"), dss):
    missing = [t for t in ds.samples if not os.path.exists(image_path(cfg.lane_images, ds.gt.cam(t)["token"]))]
    print(f"{name}: {len(ds.samples)} samples, missing lane images {len(missing)}")
