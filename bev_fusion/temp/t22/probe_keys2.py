import sys
sys.path.insert(0, "training")
import torch
from safetensors.torch import load_file
import student_encoder as S
ref = load_file(S.WEIGHTS["efficientvit_b1"])
m = S.StudentEncoder(pretrained=False)
a = list(ref)[:6]; b = list(m.backbone.state_dict())[:6]
print(a); print(b)
print(len(ref), len(m.backbone.state_dict()))
