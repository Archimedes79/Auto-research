"""bev_free - the BEV network run on frames WITHOUT the ground truth (D72): a Source that reads none and a Subject that computes no validation loss. They plug into the engine of
autoresearch/Evaluation (its Runner drives a Subject over a Source and stores the predictions) in place of bev_adapter's BEVSource / BEVSubject, and give the same predictions.

    src = free_source(built_source)         # a Source made by bev_adapter.build(...) (or bev_eval.source): the same set, frames and dataset keys, now without the ground truth
    subj = FreeSubject(built_subject)       # the BEVSubject that build() made: its model (EMA weights), config and tap recorder, now without the loss
    for ref in src.frames("scene-0003"):    # the engine does this: subject.reset() at the start of a scene, then the frames in time order (the memory is carried: D70)
        pred = subj.predict(src.load(ref), taps=("fused", "memory_conf"))

What differs from the live pair, and nothing else: `FreeSource.load` returns `Sample(reference=None)` (the dataset item is made without `gt`); `FreeSubject.predict` calls the model the way
train.forward_loss does before the loss (same ego gap, host copies, autocast, state and step) and stops there, so `info` has no `loss`; `describe()` names no ground truth. The outputs are built by
the same function as BEVSubject.predict builds them (`bev_adapter.outputs_of`) and have the same structure and dtypes:
tests/test_bev_free.py checks them bit for bit against the live subject on real frames. One network per process, like bev_adapter.build.
"""
import inspect
import os
import sys
from typing import Any, Dict, Sequence

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
AUTORESEARCH = os.path.abspath(os.path.join(ROOT, "..", "autoresearch"))
for _p in (ROOT, AUTORESEARCH):                       # the helpers of the evaluation and the contracts of Evaluation (the code snapshot of the run keeps the first place)
    if _p not in sys.path:
        sys.path.append(_p)

from Evaluation import Prediction, Sample, Subject  # noqa: E402
from evaluation import bev_adapter as A  # noqa: E402

DTS = (-0.5, 0.0, 0.5)                                # what the live path decodes: the dts of the ground truth's keys (now and +-0.5 s)
NEEDS = ("batch_ego", "supports_ego", "to_device", "amp_context")      # what every snapshot's train.forward_loss uses before the loss; `host_side` (D54) and `pin_memory` only the newer ones

_SOURCE_CLASS: Dict[str, Any] = {}


def _source_class():
    """FreeSource: made from the class bev_adapter builds its Source with (once per process), so that the set, the frames and the read-ahead are the live ones"""
    if "cls" not in _SOURCE_CLASS:
        Base = A._make_source_class()

        class FreeSource(Base):
            """the BEV Source without the ground truth: the same frames in the same order with the same dataset keys, `load` gives the network input only"""

            def __init__(self, built):
                super().__init__(built.cfg, built.ds, built.ns, workers=built.workers, control=built.control, split=built.split, evaluator=built.evaluator, set_hash=built.set_hash)
                getattr(self.ds, "dataset", self.ds).with_gt = False          # before any worker forks: the items are made without `gt`
                self._bev_subject = getattr(built, "_bev_subject", None)       # the Subject build() made travels with its Source (bev_eval.subject hands it out)

            def describe(self) -> Dict[str, Any]:
                return {"class": "FreeSource", "dataset": "nuScenes " + self.cfg.version, "split": self.split, "scenes": len(self._frames), "frames": len(self.ds),
                        "feature_cache": self.cfg.feature_cache, "control": self.control or None, "evaluator_hash": self.evaluator,
                        **({"set_hash": self.set_hash} if self.set_hash else {})}

            def load(self, ref) -> Sample:
                item = self._item(ref)
                if self.control:
                    self._perturb(item, ref)
                batch = self.fn([item])
                batch.pop("gt", None)
                if batch["tokens"][0] != ref.key:
                    raise ValueError(f"{ref.label}: the dataset item is {batch['tokens'][0]}, the frame is {ref.key}: the Source and the dataset disagree")
                return Sample(ref=ref, inputs=batch, reference=None, meta={"timestamp": item.get("timestamp"), "token": ref.key})
        _SOURCE_CLASS["cls"] = FreeSource
    return _SOURCE_CLASS["cls"]


def free_source(built):
    """a FreeSource from a Source that bev_adapter.build made (the dataset, the set, the controls are its). The built Source is consumed: it shares the dataset, which now makes its items
    without `gt`, so read the labelled frames first if you want both"""
    return _source_class()(built)


class FreeSubject(Subject):
    """the network of a BEVSubject without the loss: `predict` -> Prediction(outputs={"dt": ...}, taps, info={}); the memory is carried from frame to frame, `reset()` starts a scene"""

    def __init__(self, bev, dts: Sequence[float] = DTS):
        missing = [n for n in NEEDS if not hasattr(bev.ns.T, n)]
        if missing:
            raise ValueError(f"the code of this run ({bev.ns.code}) has no train.{', train.'.join(missing)}: it is too old for the run without ground truth; use bev_adapter's subject")
        self.b, self.dts = bev, tuple(float(d) for d in dts)
        takes_strict = "strict" in inspect.signature(bev.ns.T.batch_ego).parameters      # LossConfig.hygiene_2026_11_02: a snapshot from before it calls batch_ego without `strict`, so do we
        self._strict = {"strict": bool(getattr(getattr(bev.loss, "cfg", None), "hygiene_2026_11_02", False))} if takes_strict else {}
        self.model, self.warmup = bev.model, bev.warmup
        self.state, self.prev, self._checked = None, None, False

    @property
    def device(self):
        return self.b.device

    def reset(self) -> None:
        self.state, self.prev = None, None

    def taps(self):
        return self.b.taps()

    def describe(self) -> Dict[str, Any]:
        return {**self.b.describe(), "class": "FreeSubject"}

    def close(self) -> None:
        self.b.rec.close()                                                    # the forward hooks of the tap recorder leave the model

    def predict(self, sample: Sample, taps: Sequence[str] = ()) -> Prediction:
        import torch
        b, ns = self.b, self.b.ns
        T, model, device = ns.T, b.model, b.device
        names = set(taps)
        b.rec.arm([n for n in (names | ({"radar_feat", "radar_memory"} if "radar_aux" in names else set())) if n in b.rec.hooks])
        batch = sample.inputs
        try:
            with torch.no_grad():
                res = self._forward(batch)
                if ns.detach_tree is not None and "state" in res:
                    self.state, self.prev = ns.detach_tree(res["state"]), batch
                outputs = A.outputs_of(model, res, ns, b.refine)
                arrays = b.rec.read(0) if names else {}
                if "memory_conf" in arrays:
                    arrays["memory_conf"] = arrays["memory_conf"][0]
                if "trunk" in names:
                    arrays["trunk"] = res["features"][0].float().cpu().numpy()
                A.output_taps(model, res["out"][0.0], arrays, names, device)
                if "camera_visibility" in names and hasattr(model, "camera_visibility"):
                    arrays["camera_visibility"] = model.camera_visibility(T.to_device(batch, device), res["features"])[0, 0].float().cpu().numpy()
        except SystemExit as e:                                               # the interface checks of the model end in SystemExit: a clear error for the caller
            raise ValueError(str(e.code if e.code is not None else e)) from None
        return Prediction(outputs={"dt": outputs}, taps={k: v for k, v in arrays.items() if k in names}, info={})

    def _forward(self, batch):
        """train.forward_loss up to the model call, without the ground truth: the ego gap of the dts, the host copies, the device, autocast, the state of the previous frame"""
        from interface import check_inputs, check_outputs
        b, ns = self.b, self.b.ns
        T, model, cfg, device = ns.T, b.model, b.cfg, b.device
        ego = T.batch_ego(batch["gap_steps"], self.dts, device, **self._strict) if cfg.gap_ego and T.supports_ego(model) else None
        host = T.host_side(batch) if hasattr(T, "host_side") else None             # (a snapshot from before D54 has neither the host copies nor pin_memory: its forward_loss does neither)
        dev = T.to_device(batch, device, non_blocking=bool(getattr(cfg, "pin_memory", False)))
        if host:
            dev["host"] = host
        carried = self.state is not None and ns.clip_step is not None and self.prev is not None
        kw = {"ego": ego, **({"state": self.state, "step": ns.clip_step(self.prev, device)} if carried else {})}
        if not self._checked:
            check_inputs(dev, dts=self.dts, ego=ego, raise_on=True)
        with T.amp_context(cfg.amp, device):
            res = model(dev, dts=self.dts, **{k: v for k, v in kw.items() if v is not None})
        if not self._checked:
            check_outputs(res, self.dts, raise_on=True)
            self._checked = True
        return res
