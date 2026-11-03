"""Gate of the object head, stage 2 (plan.md 8.3, S5c): box refinement with 4 sampling points on the box faces.
On the synthetic world (inputs, network and metrics of studies/early/object_head_synth.py, rendered with ss = 2): stage 1
(BEVTrunk + ObjectHead) trained as in its --quick setting (200 training scenes, 400 iterations, batch 8), then frozen;
two refiners trained on its features (same seed, same budget): CenterPoint's fixed centre + 4 face points (control)
and the deformable points (learned offsets). Candidates in training: the decoded stage-1 boxes plus jittered GT boxes;
at evaluation the decoded boxes only. Compared on 50 held-out scenes: stage 1 alone vs + fixed vs + deformable (AP at
0.5 / 1 / 2 m, and for the matches at 1 m BEV IoU, centre error, yaw error).
Unobserved GT boxes (observed False, plan 8.4) are ignored by S5c: check_unobserved() runs first.
Pass: the deformable refinement improves the mean BEV IoU of cars over stage 1 alone. Log: temp/checks/box_refine.log
Run from the bev_fusion folder: wsl ~/miniconda3/envs/gcar/bin/python studies/early/box_refine.py
"""
import random
import sys
import time

import torch

sys.path.insert(0, "training")
sys.path.insert(0, "tests/support")
from object_head import BoxRefiner, refine_boxes, refine_candidates, refine_loss, sample_bev, boxes_to_tensor  # noqa: E402
from synthetic_world import random_scene  # noqa: E402
from object_head_synth import DEV, G, evaluate, make_model, predict, scene_inputs, show, train  # noqa: E402

LOG = "temp/checks/box_refine.log"
REF_ITERS, BS = 1500, 8


class Tee:
    def __init__(self, path):
        self.f, self.out = open(path, "w"), sys.stdout

    def write(self, s):
        self.out.write(s)
        self.f.write(s)

    def flush(self):
        self.out.flush()
        self.f.flush()


def train_refiner(ref, trunk, data, dets, iters, bs, lr=1e-3, seed=0):
    rng = random.Random(seed)
    opt = torch.optim.AdamW(ref.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=iters, pct_start=0.1)
    ref.train()
    t0 = time.time()
    for it in range(iters):
        idx = [rng.randrange(len(data)) for _ in range(bs)]
        with torch.no_grad():
            feat = trunk.features(torch.stack([data[i][0] for i in idx]).to(DEV))
        box, group, valid, gt, matched = (t.to(DEV) for t in refine_candidates([dets[i] for i in idx],
                                                                              [data[i][1] for i in idx], rng=rng))
        loss = refine_loss(ref(feat, box, group, G), gt, matched, valid, group)
        opt.zero_grad()
        loss["total"].backward()
        torch.nn.utils.clip_grad_norm_(ref.parameters(), 10.0)
        opt.step()
        sched.step()
        if it % max(1, iters // 5) == 0 or it == iters - 1:
            print(f"  iter {it:5d} loss {loss['total'].item():.3f} (reg {loss['reg'].item():.3f}, quality "
                  f"{loss['quality'].item():.3f}) {time.time() - t0:.0f} s", flush=True)


def check_unobserved():
    """plan 8.4 masks: an unobserved GT box (observed False) is never a positive or a matching target of S5c; a
    candidate at it is ignored (valid 0, matched 0, no loss), not background; an observed box is matched as before."""
    car = {"group": "car", "length": 4.5, "width": 1.8, "yaw": 0.0, "vx": 0.0, "vy": 0.0}
    gts = [[dict(car, cx=10.0, cy=0.0), dict(car, cx=30.0, cy=5.0, observed=False)]]
    dets = [[dict(car, cx=10.3, cy=0.1, score=0.9), dict(car, cx=30.2, cy=5.1, score=0.8),
             dict(car, cx=50.0, cy=-5.0, score=0.5)]]
    box, group, valid, gt, matched = refine_candidates(dets, gts, rng=random.Random(0))
    assert box.shape[1] == 4, "one jittered copy of the observed GT only"
    assert valid[0].tolist() == [1, 0, 1, 1] and matched[0].tolist() == [1, 0, 0, 1], (valid, matched)
    ref = BoxRefiner(4)
    base = refine_loss(ref(torch.randn(1, 4, G.nx, G.ny), box, group, G), gt, matched, valid, group)["total"]
    box2 = box.clone()
    box2[0, 1, :2] += 0.7                                        # moving the ignored candidate changes nothing
    torch.manual_seed(0)
    feat = torch.randn(1, 4, G.nx, G.ny)
    l1 = refine_loss(ref(feat, box, group, G), gt, matched, valid, group)["total"]
    l2 = refine_loss(ref(feat, box2, group, G), gt, matched, valid, group)["total"]
    assert torch.isfinite(base) and abs(l1.item() - l2.item()) < 1e-6
    print("unobserved check ok: no jittered copy, the candidate at an unobserved box is ignored (valid 0, matched 0, "
          "no loss), observed boxes matched")


@torch.no_grad()
def predict_refined(trunk, ref, data, dets):
    ref.eval()
    out, moved = [], []
    for i in range(0, len(data), 8):
        feat = trunk.features(torch.stack([d[0] for d in data[i:i + 8]]).to(DEV))
        out.extend(refine_boxes(ref, feat, dets[i:i + 8], G))
        box, group, valid = boxes_to_tensor(dets[i:i + 8], device=DEV)
        rel = ref.points(box, group, sample_bev(feat, box[..., None, :2], G)[..., 0, :])[0]
        base = ref.points(box, group)[0]
        moved.append(((rel - base).norm(dim=-1).mean(-1) * valid).sum().item() / max(1.0, valid.sum().item()))
    return out, sum(moved) / len(moved)


@torch.no_grad()
def ms_per_frame(trunk, ref, data, dets, n=20):
    feat = trunk.features(data[0][0][None].to(DEV))
    for _ in range(3):
        refine_boxes(ref, feat, dets[:1], G)
    if DEV == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n):
        refine_boxes(ref, feat, dets[:1], G)
    if DEV == "cuda":
        torch.cuda.synchronize()
    return 1000 * (time.time() - t0) / n


if __name__ == "__main__":
    sys.stdout = Tee(LOG)
    torch.manual_seed(0)
    random.seed(0)
    print(f"device {DEV}")
    check_unobserved()
    t0 = time.time()
    rng = random.Random(2)
    train_set = [scene_inputs(random_scene(rng), ss=2) for _ in range(200)]
    rng = random.Random(999)
    test_set = [scene_inputs(random_scene(rng), ss=2) for _ in range(50)]
    print(f"200 training scenes, 50 held-out scenes (rendered in {time.time() - t0:.0f} s)")
    print("(1) stage 1: BEVTrunk + ObjectHead, 400 iterations")
    trunk, head = make_model()
    train(trunk, head, train_set, 400, BS)
    for p in list(trunk.parameters()) + list(head.parameters()):
        p.requires_grad_(False)
    trunk.eval()
    dets_train, dets_test = predict(trunk, head, train_set), predict(trunk, head, test_set)
    gts_test = [d[1] for d in test_set]
    results = {"stage 1": evaluate(dets_test, gts_test)}
    show(results["stage 1"], "    held-out, stage 1")
    cin = trunk.img_bev_encoder_neck.out_channels
    extra = {}
    for name, deformable in (("+ fixed 5 points", False), ("+ deformable", True)):
        print(f"(2) refiner {name}, {REF_ITERS} iterations, stage 1 frozen")
        torch.manual_seed(1)
        ref = BoxRefiner(cin, deformable=deformable).to(DEV)
        train_refiner(ref, trunk, train_set, dets_train, REF_ITERS, BS, seed=1)
        dets_ref, moved = predict_refined(trunk, ref, test_set, dets_test)
        results[name] = evaluate(dets_ref, gts_test)
        extra[name] = (moved, ms_per_frame(trunk, ref, test_set, dets_test))
        show(results[name], f"    held-out, stage 1 {name}")
    print("\nheld-out synthetic scenes (50), matches at 1 m; yaw in deg; point offset = mean |learned offset| (m)")
    cols = ("car IoU", "car centre", "car yaw mod pi", "AP car @0.5", "AP car @1", "truck IoU", "AP truck @0.5",
            "ped IoU", "AP ped @0.5", "pt offset", "ms (busy GPU)")
    print(f"{'arm':18s} " + " ".join(f"{c:>14s}" for c in cols))
    for name, r in results.items():
        mc, mt, mp = r.get("match car", {}), r.get("match truck", {}), r.get("match pedestrian", {})
        mv, ms = extra.get(name, (float("nan"), float("nan")))
        vals = (mc.get("IoU", float("nan")), mc.get("centre", float("nan")), mc.get("yaw mod pi", float("nan")),
                r["AP car @0.5"], r["AP car @1.0"], mt.get("IoU", float("nan")), r["AP truck @0.5"],
                mp.get("IoU", float("nan")), r["AP pedestrian @0.5"], mv, ms)
        print(f"{name:18s} " + " ".join(f"{v:14.3f}" for v in vals))
    i1, i2 = results["stage 1"]["match car"]["IoU"], results["+ deformable"]["match car"]["IoU"]
    ok = i2 > i1
    print(f"\ngate: deformable car IoU {i2:.3f} vs stage 1 {i1:.3f} -> {'PASS' if ok else 'FAIL'}")
    sys.stdout.flush()
    sys.exit(0 if ok else 1)
