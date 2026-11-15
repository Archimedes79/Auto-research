"""Scratch: fetch the timm EfficientViT-B1 (r256, ImageNet-1k) weights into models/pretrained/imagenet/ and print file, size, sha256, revision.
Run from bev_fusion/: wsl ~/miniconda3/envs/gcar/bin/python temp/t22/fetch_evit_b1.py"""
import hashlib
import os

from huggingface_hub import HfApi, hf_hub_download

REPO = "timm/efficientvit_b1.r256_in1k"
OUT = "models/pretrained/imagenet/efficientvit_b1_r256_in1k"
info = HfApi().model_info(REPO)
print("license tags:", [t for t in info.tags if "license" in t], "revision:", info.sha)
print("files:", [s.rfilename for s in info.siblings])
os.makedirs(OUT, exist_ok=True)
fn = "model.safetensors" if any(s.rfilename == "model.safetensors" for s in info.siblings) else "pytorch_model.bin"
p = hf_hub_download(REPO, fn, local_dir=OUT, revision=info.sha)
h = hashlib.sha256(open(p, "rb").read()).hexdigest()
print("file:", p, os.path.getsize(p), "bytes")
print("sha256:", h)
with open(os.path.join(OUT, "SOURCE.txt"), "w") as f:
    f.write(f"https://huggingface.co/{REPO} revision {info.sha}\nlicense tags: {[t for t in info.tags if 'license' in t]}\n{h} *{fn}\n"
            "downloaded 2026-11-04 on Mirko's word (\"do what you propose\"); timm weights of the MIT Han Lab EfficientViT-B1 (arXiv 2205.14756)\n")
