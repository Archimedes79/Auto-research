"""Gate of experiment t7 (lane branch in the whole chain), nuScenes mini, CPU:

1. the lane image cache equals the dataset's own image (resize + crop) within JPEG error
2. the cached dataset carries the current frame as uint8 "lane_img", collate_cached stacks it
3. zero start: the chain with the lane branch gives EXACTLY the reference chain's outputs (same weights elsewhere)
4. gradients: the zero-initialised mix gets a gradient at once; after one step the gradient reaches the branch's image layers
5. mirror augmentation: for a flipped sample the branch's contribution to the camera BEV is the mirror of the unflipped one

Run (from the bev_fusion folder): python tests/test_lane_branch_chain.py [res4|rgb]
"""
import os
import sys
import tempfile
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "training"))
os.chdir(os.path.join(HERE, ".."))

import lane_branch as LB                                                  # noqa: E402
from feature_cache import CachedBEVFusionModel, CachedFrontRadarDataset, build, collate_cached   # noqa: E402
from model import ModelConfig                                             # noqa: E402
from nuscenes_dataset import FrontRadarDataset                            # noqa: E402


def main(kind="res4"):
    torch.manual_seed(0)
    t0 = time.time()
    root = os.path.expanduser("~/data/nuscenes")
    ds = FrontRadarDataset(root, "v1.0-mini", n_cam=3, cam_stride=2, with_gt=False)
    tokens = [ds.samples[10], ds.samples[200]]
    with tempfile.TemporaryDirectory() as feat_dir, tempfile.TemporaryDirectory() as img_dir:
        store = build(ds, feat_dir, tokens, device="cpu", batch=4, verbose=False)
        LB._DS = ds
        LB._fill((img_dir, tokens))
        # 1. cache = dataset image within JPEG error
        for t in tokens:
            sd = ds.gt.cam(t)
            ref = torch.tensor(ds._image_uint8(sd)).permute(2, 0, 1).float()
            got = LB.load_image(img_dir, sd["token"]).float()
            err = (got - ref).abs().mean().item()
            assert got.shape == ref.shape == (3, 256, 704) and err < 3.0, (got.shape, err)
        print(f"1. image cache = dataset image within JPEG error (mean abs {err:.2f} of 255): ok")
        # 2. dataset + collate
        cds = CachedFrontRadarDataset(root, "v1.0-mini", store=store, n_cam=3, cam_stride=2, with_gt=False,
                                      lane_images=img_dir)
        idx = [ds.samples.index(t) for t in tokens]
        batch = collate_cached([cds[i] for i in idx])
        assert batch["lane_img"].dtype == torch.uint8 and batch["lane_img"].shape == (2, 3, 256, 704)
        print("2. cached dataset carries lane_img (uint8, current frame), collate stacks it: ok")
        # 3. zero start = reference outputs exactly
        ref = CachedBEVFusionModel(ModelConfig()).eval()
        lan = CachedBEVFusionModel(ModelConfig(lane_branch=kind)).eval()
        rep = lan.load_state_dict(ref.state_dict(), strict=False)
        assert not rep.unexpected_keys and all(k.startswith("lane.") for k in rep.missing_keys), rep
        with torch.no_grad():
            o_ref, o_lan = ref(batch)["out"][0.0], lan(batch)["out"][0.0]
        d = max((o_ref["sem"] - o_lan["sem"]).abs().max().item(),
                max((o_ref["obj"][k] - o_lan["obj"][k]).abs().max().item() for k in o_ref["obj"]))
        assert d == 0.0, f"zero start differs from the reference by {d}"
        print(f"3. zero start: outputs equal the reference chain exactly (max diff {d}): ok")
        # 4. gradients
        lan.train()
        lan(batch)["out"][0.0]["sem"].float().pow(2).mean().backward()
        assert lan.lane.mix.weight.grad is not None and lan.lane.mix.weight.grad.abs().sum() > 0, "mix gets no gradient"
        first = lan.lane.reduce[0].weight.grad
        assert first is None or first.abs().sum() == 0, "zero start: nothing behind the mix gets a gradient at step 0"
        with torch.no_grad():
            lan.lane.mix.weight.add_(-0.1 * lan.lane.mix.weight.grad)
        lan.zero_grad()
        lan(batch)["out"][0.0]["sem"].float().pow(2).mean().backward()
        g = lan.lane.reduce[0].weight.grad
        assert g is not None and g.abs().sum() > 0, "after one step the gradient must reach the branch"
        if kind == "rgb":
            assert lan.lane.features.net[0].weight.grad.abs().sum() > 0
        else:
            assert all(p.grad is None for p in lan.lane.features.parameters()), "res4 features must stay frozen"
        print("4. gradient: mix at step 0, the branch after one step, frozen res4 stays frozen: ok")
        # 5. mirror
        lan.eval()
        torch.nn.init.normal_(lan.lane.mix.weight, std=0.1)
        cams = []
        hook = lan.fusion.register_forward_pre_hook(lambda m, a: cams.append(a[0].detach().clone()))
        twin = CachedBEVFusionModel(ModelConfig(lane_branch=kind)).eval()
        twin.load_state_dict(lan.state_dict())
        torch.nn.init.zeros_(twin.lane.mix.weight)
        hook2 = twin.fusion.register_forward_pre_hook(lambda m, a: cams.append(a[0].detach().clone()))
        deltas = {}
        with torch.no_grad():
            for fl in ((False, False), (True, False)):
                batch["flip"] = torch.tensor(fl)
                cams.clear()
                lan(batch)
                twin(batch)
                deltas[fl] = cams[0] - cams[1]                       # the branch's contribution alone
        hook.remove()
        hook2.remove()
        a, b = deltas[(False, False)], deltas[(True, False)]
        assert a.abs().max() > 0
        assert torch.allclose(b[0], a[0].flip(-1), atol=1e-6) and torch.allclose(b[1], a[1], atol=1e-6), "mirror"
        print("5. mirror: a flipped sample gets the mirrored lane contribution, the other one is unchanged: ok")
    print(f"lane branch chain gate ({kind}) passed in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "res4")
