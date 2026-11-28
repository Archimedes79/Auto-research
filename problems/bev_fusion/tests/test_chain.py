"""The network's entry for the trainer and the evaluation (net/chain.py), the terms only a training step has (net/loss/train_terms.py) and the clip helpers of
net/prep/batches.py; CPU, the tiny model on synthetic keyframes, about a minute.

  1  read_config takes the network's keys of a stored config and leaves the engine's and the trainer's out; the trainer's config holds every field of
     ChainConfig with the same default; the base config loads through both; an option that left the code loads at its off value, is left out of the model
     and the loss and does not stand in the way of a resume, and is refused at any other value
  2  load_weights gives the weights of the engine's own loader followed by EMA.apply (the reference), bit for bit, in eval mode; a checkpoint that lacks a key
     of the model or holds one the model does not have (in the trained state or in the average) is refused
  3  forward (no ground truth) gives the predictions of forward_loss bit for bit; the state carried from the keyframe before changes the prediction
  4  the train-only terms: total = task loss + weight x term for the consistency and the radar side loss, their gradients reach the motion block and the radar
     decoder; inside the hold nothing is computed; in validation the total is the task loss and the radar side logs the share it keeps
  5  known answers: the consistency ramp, the side-loss and anchor weights, the ego motion of a gap on a line and on an arc, the step between two keyframes,
     the validation subset

    python -P -m bev_fusion.tests.test_chain      (from the problem folder, with its module path)
"""
import dataclasses
import json
import math
import os
import tempfile

import torch

from bev_fusion.net import chain, netpath
from bev_fusion.net.loss.radar_loss import aux_weight
from bev_fusion.net.loss.student_loss import anchor_weight
from bev_fusion.net.loss.train_terms import consistency_weight
from bev_fusion.net.prep import splits
from bev_fusion.net.prep.batches import clip_step, gap_ego_motion, val_subset
from bev_fusion.net.prep.feature_cache import collate_cached
from bev_fusion.tests.support.synthetic_batch import keyframe
from bev_fusion.training import train as T
from trainkit import checkpoint as tk_ckpt
from trainkit.ema import EMA

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
MODEL = dict(pretrained="", trunk_pretrained="", oversample=1, c_cam=8, c_rad=64, c_fuse=16, trunk_width=0.25, radar_kind="delta_slots", motion_block=True)
CONFIG = dict(feature_cache="synthetic", gt_cache="", seq_len=2, consistency=0.5, consistency_warmup=(0.0, 0.5), radar_aux=1.0, model=MODEL)
DTS = (-0.5, 0.0, 0.5)
torch.set_num_threads(2)


def raises(fn, error, text):
    try:
        fn()
    except error as e:
        return text in str(e)
    return False


def test_config():
    fields = {f.name: f for f in dataclasses.fields(chain.ChainConfig)}
    train_fields = {f.name: f for f in dataclasses.fields(T.TrainConfig)}
    assert set(fields) <= set(train_fields) and all(train_fields[k].default == f.default for k, f in fields.items()), "the trainer's config holds the network's fields"
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "config.json")
        json.dump({"_about": "a comment", "steps": 5, "workers": 3, "n_cam": 2, "model": {"c_cam": 8}}, open(path, "w"))
        cfg = chain.read_config(path)
        assert cfg.n_cam == 2 and cfg.model == {"c_cam": 8} and cfg.gap_ego is True and not hasattr(cfg, "steps") and not hasattr(cfg, "workers")
        json.dump({"n_cam": 2, "no_such_key": 1}, open(path, "w"))
        assert raises(lambda: T.read_config(path), TypeError, "no_such_key"), "the trainer refuses an unknown key"
        # an option that left the code, named at its off value (stored configs do): both readers load the config, the top-level key is dropped, a key of
        # model / loss stays in its dict and is left out where the model and the loss are built; at any other value both readers refuse
        stored = {"uwso_T": 5.0, "steps": 5, "model": {"sync_free": False, "fusion": "concat", "refine": False, "c_cam": 8}, "loss": {"sync_free": False, "line_bce": 0.0}}
        json.dump(stored, open(path, "w"))
        cfg, whole = chain.read_config(path), T.read_config(path)
        assert cfg.model == stored["model"] == whole.model and cfg.loss == stored["loss"] and not hasattr(whole, "uwso_T")
        assert chain.live("model", cfg.model) == {"c_cam": 8} and chain.live("loss", cfg.loss) == {}
        json.dump({"uwso_T": 2.0, "model": {"fusion": "gate"}}, open(path, "w"))
        assert all(raises(lambda: read(path), ValueError, text) for read in (chain.read_config, T.read_config) for text in ("model.fusion = 'gate'", "uwso_T = 2.0"))
        # a resume of a run of before (the engine compares the config its checkpoint recorded with today's): the removed keys do not stand in the way,
        # every other difference still does
        before = {**dataclasses.asdict(whole), **{k: v for k, v in stored.items() if k in chain.REMOVED[""]}}
        assert set(chain.REMOVED[""]) <= set(before) and before["model"] == stored["model"]
        tk_ckpt.check_resume(before, whole, "run")
        assert raises(lambda: tk_ckpt.check_resume({**before, "steps": 6}, whole, "run"), SystemExit, "['steps']")
        assert raises(lambda: tk_ckpt.check_resume({**before, "model": {"c_cam": 16}}, whole, "run"), SystemExit, "['model']")
    assert chain.build(chain.ChainConfig(**dict(CONFIG, model=dict(MODEL, refine=False), loss={"w_refine": 1.0})))[0].cfg.c_cam == 8, "built without the removed keys"
    assert raises(lambda: chain.build(chain.ChainConfig(**dict(CONFIG, model=dict(MODEL, refine=True)))), ValueError, "model.refine = True"), "a config made in code too"
    for removed_kind in (dict(model=dict(MODEL, radar_kind="pillars")), dict(balancing="uwso")):
        assert raises(lambda: chain.build(chain.ChainConfig(**dict(CONFIG, **removed_kind))), ValueError, "removed in v2"), removed_kind
    base = os.path.join(ROOT, "training", "configs", "base.json")
    net_side, whole = chain.read_config(base), T.read_config(base)
    assert all(getattr(net_side, k) == getattr(whole, k) for k in fields) and whole.steps == 1500 and len(whole.train_scenes) == 428 and whole.model["lane_branch"] == "res4"
    # the scene lists are in one place: the base config names its split and both readers fill the lists in; a stored config that carries them inline (the
    # measured candidates) loads to the same scenes
    large, e27 = splits.load("large"), os.path.join(ROOT, "experiments", "E027_t21_measured", "runs", "E27.t21_s0", "config.json")
    named, inline = json.load(open(base)), json.load(open(e27))
    assert named["split"] == "large" and named["train_scenes"] == [] == named["val_scenes"] and "split" not in inline and len(inline["train_scenes"]) == 428
    for cfg in (net_side, whole, chain.read_config(e27), T.read_config(e27)):
        assert list(cfg.train_scenes) == large["train"] and list(cfg.val_scenes) == large["val"]
    # the raw data (netpath.nuscenes_root): the view of the data root, the older link where the view is missing; the data_root the stored configs name is
    # that link, the same folder
    kept = os.environ.get("RESEARCH_DATA")
    with tempfile.TemporaryDirectory() as tmp:
        os.makedirs(os.path.join(tmp, "nuscenes"))
        os.environ["RESEARCH_DATA"] = tmp
        try:
            assert netpath.nuscenes_root() == os.path.join(tmp, "nuscenes") and netpath.nuscenes_root(mini=True) == os.path.expanduser("~/data/nuscenes")
        finally:
            os.environ.pop("RESEARCH_DATA") if kept is None else os.environ.update(RESEARCH_DATA=kept)
    assert os.path.realpath(os.path.expanduser(whole.data_root)) == os.path.realpath(netpath.nuscenes_root()), "the base config reads the machine's view of nuScenes"
    return "config"


def test_weights():
    cfg = chain.ChainConfig(**CONFIG)
    torch.manual_seed(0)
    model, loss, _ = chain.build(cfg)
    ema = EMA(model, 0.5)
    with torch.no_grad():                                        # the trained weights move away from the average: the two must not be confused
        for p in model.parameters():
            p.add_(0.01 * torch.randn_like(p))
        loss.task.s.add_(0.25)
    ema.update(model, 100)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "ckpt_00000007.pt")
        tk_ckpt.save_checkpoint(path, model, loss, opt, ema, 7, cfg, slim=True, external=chain.external(model))
        ref_model, ref_loss, _ = chain.build(cfg)                # the reference: the engine's loader, then the average applied
        ref_ema = EMA(ref_model, 0.5)
        tk_ckpt.load_checkpoint(path, ref_model, ref_loss, None, ref_ema, external=chain.external(ref_model))
        ref_ema.apply(ref_model)
        new_model, new_loss, _ = chain.build(cfg)
        assert chain.load_weights(new_model, new_loss, path) == 7 and not new_model.training
        ref, new = ref_model.state_dict(), new_model.state_dict()
        assert set(ref) == set(new) and all(torch.equal(ref[k], new[k]) for k in ref), "load_weights = the engine's loader + EMA.apply"
        assert all(torch.equal(v, new_loss.state_dict()[k]) for k, v in ref_loss.state_dict().items()) and float(new_loss.task.s[0]) == float(loss.task.s[0])
        some = next(k for k in ema.shadow if k.startswith("fusion."))
        assert torch.equal(new[some], ema.shadow[some]) and not torch.equal(new[some], model.state_dict()[some]), "the average is evaluated, not the last weights"
        # a checkpoint of another code version or config is refused: a key the model lacks in the trained state or in the average, a key the checkpoint lacks
        ck = torch.load(path, weights_only=False)
        for part, key, text in (("state", "model.gone.weight", "1 keys the model does not have"), ("ema", "gone.weight", "the EMA holds 1 keys")):
            torch.save({**ck, part: {**ck[part], key: torch.zeros(1)}}, path)
            assert raises(lambda: chain.load_weights(new_model, new_loss, path), RuntimeError, text), (part, "a key the model does not have is refused")
        del ck["state"]["model." + some]
        torch.save(ck, path)
        assert raises(lambda: chain.load_weights(new_model, new_loss, path), RuntimeError, "1 keys it lacks"), "a checkpoint that lacks a key is refused"
    return "weights"


def test_forward_and_terms():
    cfg = chain.ChainConfig(**CONFIG)
    torch.manual_seed(0)
    model, loss, _ = chain.build(cfg)
    mcfg = model.cfg
    b0, b1 = (collate_cached([keyframe(mcfg, i)]) for i in (0, 1))
    free = {k: v for k, v in b1.items() if k != "gt"}
    model.eval()
    with torch.no_grad():
        out0, res0, _ = chain.forward_loss(model, loss, b0, 1.0, "cpu", cfg, train_terms=False)
        step = clip_step(b0, "cpu")
        out1, res1, gts = chain.forward_loss(model, loss, b1, 1.0, "cpu", cfg, train_terms=False, state=res0["state"], step=step)
        res_free, _ = chain.forward(model, free, DTS, "cpu", cfg, state=res0["state"], step=step)
        res_cold, _ = chain.forward(model, free, DTS, "cpu", cfg)
    assert tuple(sorted(gts)) == DTS and set(res_free["out"]) == set(DTS)
    for dt in DTS:
        a, b = res1["out"][dt], res_free["out"][dt]
        assert torch.equal(a["sem"], b["sem"]) and all(torch.equal(a["obj"][k], b["obj"][k]) for k in a["obj"]) and torch.equal(a["mass"], b["mass"]), dt
    assert float((res_free["out"][0.0]["sem"] - res_cold["out"][0.0]["sem"]).abs().max()) > 1e-6, "the carried memory changes the prediction"
    assert float(out1["total"]) == float(out1["total_main"]) and "consistency" not in out1 and "radar_aux" not in out1, "validation: the total is the task loss"
    assert 0.0 <= float(out1["radar_kept"]) <= 1.0
    # training: the terms enter the total with their weights, and their gradients reach what they are for
    model.train()
    out, _, _ = chain.forward_loss(model, loss, b1, 0.9, "cpu", cfg)
    want = float(out["total_main"]) + consistency_weight(cfg, 0.9) * float(out["consistency"]) + aux_weight(0.9, 1.0, cfg.radar_aux_floor, cfg.radar_aux_half) * float(out["radar_aux"])
    assert consistency_weight(cfg, 0.9) == 0.5 and abs(float(out["total"]) - want) < 1e-5 * max(1.0, abs(want)), (float(out["total"]), want)
    grad = lambda term, module: sum(float(g.abs().sum()) for g in torch.autograd.grad(term, list(module.parameters()), retain_graph=True, allow_unused=True) if g is not None)      # noqa: E731
    assert grad(out["consistency"], model.motion_block) > 0, "the consistency term reaches the motion block"
    assert grad(out["radar_aux"], model.radar.decoder) > 0, "the side loss reaches the radar decoder"
    held = dataclasses.replace(cfg, consistency_warmup=(0.5, 0.1))
    out_h, _, _ = chain.forward_loss(model, loss, b1, 0.3, "cpu", held)
    assert "consistency" not in out_h and "radar_aux" in out_h, "inside the hold the consistency is not computed"
    return "forward, train terms"


def test_known_answers():
    cfg = chain.ChainConfig(consistency=1.0, consistency_warmup=(0.5, 0.1))
    assert all(abs(consistency_weight(cfg, f) - w) < 1e-9 for f, w in ((0.3, 0.0), (0.5, 0.0), (0.55, 0.5), (0.6, 1.0), (0.9, 1.0)))
    assert abs(aux_weight(0.0, 1.0, 0.05, 0.2) - 1.0) < 1e-12 and abs(aux_weight(0.2, 1.0, 0.05, 0.2) - 0.525) < 1e-12 and abs(aux_weight(40.0, 1.0, 0.05, 0.2) - 0.05) < 1e-9
    assert all(abs(anchor_weight(f, 2.0, 0.1, 0.5) - w) < 1e-12 for f, w in ((0.0, 2.0), (0.25, 1.1), (0.5, 0.2), (0.9, 0.2)))
    T_, v, w = gap_ego_motion([(1 / 12, 10.0, 0.0)] * 6)                      # a straight line, forward and backward
    assert abs(T_ - 0.5) < 1e-9 and abs(v - 10) < 1e-9 and abs(w) < 1e-9, (T_, v, w)
    T_, v, w = gap_ego_motion([(-1 / 12, 10.0, 0.0)] * 6)
    assert abs(T_ + 0.5) < 1e-9 and abs(v - 10) < 1e-9, (T_, v, w)
    for sgn in (1, -1):                                                      # an arc: chord = v / omega sin(omega T), and the splat's lateral term matches it
        T_, v, w = gap_ego_motion([(sgn / 12, 10.0, 0.2)] * 6)
        assert abs(v * T_ - sgn * 50 * math.sin(0.1)) < 2e-3 and abs(w - 0.2) < 1e-9, (sgn, T_, v, w)
        assert abs(0.5 * v * w * T_ * T_ - 50 * (1 - math.cos(0.1))) < 2e-3
    assert gap_ego_motion([]) == (0.0, 0.0, 0.0)
    gaps = [{"next": [(1 / 12, 8.0, 0.0)] * 6}, {"next": [(0.45 / 6, 6.0, 0.0)] * 6}]
    dt, v, w = clip_step({"gap_steps": gaps}, "cpu")
    assert torch.allclose(dt, torch.tensor([0.5, 0.45])) and torch.allclose(v, torch.tensor([8.0, 6.0])) and float(w.abs().max()) == 0.0, "the step between two keyframes is the true gap"
    sub = val_subset(list(range(100)), 10)
    assert len(sub) == 10 and sub[0] == 0 and sub[-1] == 99 and val_subset(list(range(5)), 10) == list(range(5))
    return "known answers"


if __name__ == "__main__":
    done = [test() for test in (test_config, test_known_answers, test_weights, test_forward_and_terms)]
    print("test_chain passed: " + ", ".join(done))
