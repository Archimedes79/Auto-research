import os, time, torch, timm
torch.set_num_threads(2)
p = "models/pretrained/imagenet/efficientvit_b1_r256_in1k/model.safetensors"
print(os.path.exists(p), os.path.exists("models/pretrained/imagenet/resnet18-f37072fd.pth"))
print(os.listdir("models/pretrained/imagenet"))
t = time.time()
m = timm.create_model("efficientvit_b1.r256_in1k", pretrained=True, features_only=True, pretrained_cfg_overlay=dict(file=p))
print("load", time.time() - t, type(m).__name__, m.feature_info.channels(), m.feature_info.reduction())
x = torch.randn(1, 3, 256, 704)
with torch.no_grad():
    fs = m(x)
print([tuple(f.shape) for f in fs])
print(sum(q.numel() for q in m.parameters()) / 1e6, "M params")
print([n for n, _ in m.named_children()])
bn = sum(1 for mm in m.modules() if isinstance(mm, torch.nn.modules.batchnorm._BatchNorm))
print("bn modules", bn)
print(m.pretrained_cfg.get("mean"), m.pretrained_cfg.get("std"))
