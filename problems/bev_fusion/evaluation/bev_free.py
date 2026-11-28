"""bev_free - the BEV network run on frames WITHOUT the ground truth: a Source that reads none and a Subject that computes no validation loss. They plug into
the engine of
autoresearch/Evaluation (its Runner drives a Subject over a Source and stores the predictions) in place of bev_adapter's BEVSource / BEVSubject, and give the
same predictions.

    src = free_source(built_source)         # a Source made by bev_adapter.build(...) (or bev_eval.source): the same set, frames and dataset keys, now without
    the ground truth
    subj = FreeSubject(built_subject)       # the BEVSubject that build() made: its model (EMA weights), config and tap recorder, now without the loss
    for ref in src.frames("scene-0003"):    # the engine does this: subject.reset() at the start of a scene, then the frames in time order (the memory is
    carried)
        pred = subj.predict(src.load(ref), taps=("fused", "memory_conf"))

What differs from the labelled pair, and nothing else: `FreeSource.load` returns `Sample(reference=None)` (the dataset item is made without `gt`);
`FreeSubject.predict` runs
`chain.forward`, the model call of `chain.forward_loss` without the loss, so `info` has no `loss`; `describe()` names no ground truth. The outputs and the taps
are built
by the functions BEVSubject.predict uses (`bev_adapter.outputs_of`, `read_taps`). One network per process, like bev_adapter.build.
"""
from typing import Any, Dict, Sequence

import torch

from net import chain
from Evaluation import Prediction, Sample, Subject
from evaluation import bev_adapter as A

DTS = (-0.5, 0.0, 0.5)                                # what the labelled path decodes: the dts of the ground truth's keys (now and +-0.5 s)


class FreeSource(A.BEVSource):
    """the BEV Source without the ground truth: the same frames in the same order with the same dataset keys, `load` gives the network input only"""

    def __init__(self, built):
        super().__init__(built.cfg, built.ds, workers=built.workers, control=built.control, split=built.split, evaluator=built.evaluator, set_hash=built.set_hash)
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


def free_source(built):
    """a FreeSource from a Source that bev_adapter.build made (the dataset, the set, the controls are its). The built Source is consumed: it shares the dataset,
    which now makes its items
    without `gt`, so read the labelled frames first if you want both"""
    return FreeSource(built)


class FreeSubject(Subject):
    """the network of a BEVSubject without the loss: `predict` -> Prediction(outputs={"dt": ...}, taps, info={}); the memory is carried from frame to frame,
    `reset()` starts a scene"""

    def __init__(self, bev, dts: Sequence[float] = DTS):
        self.b, self.dts = bev, tuple(float(d) for d in dts)
        self.model, self.warmup = bev.model, bev.warmup

    @property
    def device(self):
        return self.b.device

    def reset(self) -> None:
        self.b.reset()

    def taps(self):
        return self.b.taps()

    def describe(self) -> Dict[str, Any]:
        return {**self.b.describe(), "class": "FreeSubject"}

    def close(self) -> None:
        self.b.rec.close()                                                    # the forward hooks of the tap recorder leave the model

    def predict(self, sample: Sample, taps: Sequence[str] = ()) -> Prediction:
        b = self.b
        names = b.arm(taps)
        batch = sample.inputs
        try:
            with torch.no_grad():
                res, _ = chain.forward(b.model, batch, self.dts, b.device, b.cfg, strict=b.loss.cfg.hygiene_2026_11_02, **b.carried())
                b.state, b.prev = res["state"], batch
                outputs = A.outputs_of(b.model, res, b.refine)
                arrays = A.read_taps(b, res, batch, names)
        # the interface checks of the model end in SystemExit: a clear error for the caller
        except SystemExit as e:
            raise ValueError(str(e.code if e.code is not None else e)) from None
        return Prediction(outputs={"dt": outputs}, taps=arrays, info={})
