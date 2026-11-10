"""Exponential moving average of the weights and the buffers; the evaluation runs under it."""
import contextlib

import torch


class EMA:
    """Exponential moving average of the trainable weights and the buffers (BatchNorm statistics)."""

    def __init__(self, model, decay, skip=()):
        self.decay = decay
        self.skip = frozenset(skip)                              # state keys that a freeze plan fixed: their average stays the value itself (freeze.frozen_keys)
        self.shadow = {k: v.detach().clone() for k, v in self._tensors(model).items()}
        self.backup = None

    @staticmethod
    def _tensors(model):
        return {k: v for k, v in model.state_dict().items() if v.dtype.is_floating_point}

    def update(self, model, step):
        d = min(self.decay, (1 + step) / (10 + step))
        live = [(k, v) for k, v in self._tensors(model).items() if k not in self.skip]
        full = [(k, v) for k, v in live if v.dtype in (torch.float32, torch.float64)]
        if full:                                                 # two multi-tensor calls instead of two kernels per state tensor (218 float tensors of the t10 model: 436 launches of the ~20000 per step)
            shadow = [self.shadow[k] for k, _ in full]
            torch._foreach_mul_(shadow, d)
            torch._foreach_add_(shadow, [v.detach() for _, v in full], alpha=1 - d)
        for k, v in live:                                        # a multi-tensor call rounds the scalar to the tensor's dtype (bf16 / fp16 state would be averaged with 0.98828125 for 0.99): those take the loop
            if v.dtype not in (torch.float32, torch.float64):
                self.shadow[k].mul_(d).add_(v.detach(), alpha=1 - d)

    def apply(self, model):
        self.backup = {k: v.detach().clone() for k, v in self._tensors(model).items()}
        model.load_state_dict({**model.state_dict(), **self.shadow}, strict=True)

    def restore(self, model):
        model.load_state_dict({**model.state_dict(), **self.backup}, strict=True)
        self.backup = None

    @contextlib.contextmanager
    def applied(self, model):
        """`with ema.applied(model):` the model holds the averaged weights in eval mode; on leaving, the training weights are back and the model trains again"""
        self.apply(model)
        model.eval()
        try:
            yield
        finally:
            self.restore(model)
            model.train()
