import json, sys
sys.path.insert(0, "training")
from train import TrainConfig
from model import ModelConfig
from losses import LossConfig
d = json.load(open("training/configs/t3_radar_only.json"))
fields = set(TrainConfig.__dataclass_fields__)
print("unknown train keys:", sorted(set(d) - fields))
print("unknown model keys:", sorted(set(d["model"]) - set(ModelConfig.__dataclass_fields__)))
print("unknown loss keys:", sorted(set(d["loss"]) - set(LossConfig.__dataclass_fields__)))
print("steps", d["steps"], "batch", d["batch_size"], "accum", d["accum"], "seq_len", d["seq_len"], "n scenes", len(d["train_scenes"]), len(d.get("val_scenes", [])))
