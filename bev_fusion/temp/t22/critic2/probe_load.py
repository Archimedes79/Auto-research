import os, sys, torch
sys.path[:0] = ["training", "../autoresearch"]
torch.set_num_threads(2)
import student_encoder as S
from trainkit.checkpoint import warm_start
run = "runs/t22_A1_s0"
p = S.newest_checkpoint(run)
print("newest", p)
ck = torch.load(p, map_location="cpu", weights_only=False)
print("backbone", ck["cfg"]["backbone"], "step", ck["step"], "keys state", len(ck["state"]), "ema", len(ck["ema"]))
a = S.StudentEncoder(ck["cfg"]["backbone"], pretrained=False)
fresh_old = warm_start(p, a, torch.nn.Module(), ck=ck)          # the old eval_only path
b = S.StudentEncoder(ck["cfg"]["backbone"], pretrained=False)
S.load_trained(b, p, ck=ck)                                      # the new path (strict)
print("old fresh keys:", fresh_old[:5], len(fresh_old))
print("same weights:", all(torch.equal(a.state_dict()[k], b.state_dict()[k]) for k in a.state_dict()))
# the weights of the chain model from student_init=run folder
import model as M
from feature_cache import CachedBEVFusionModel
import dataclasses
sys.path.insert(0, "tests")
from test_slots import BASE
cfg = M.ModelConfig(**dict(BASE, camera_features="dual", camera_student="evit_b1", student_init=run, student_trainable=False))
m = CachedBEVFusionModel(cfg)
print("chain student equals eval_only weights:", all(torch.equal(m.student.state_dict()[k], b.state_dict()[k]) for k in b.state_dict()))
