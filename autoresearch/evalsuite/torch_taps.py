"""Optional helper for torch networks: record intermediate tensors with forward hooks, by module path.

    rec = TapRecorder(model, {"lift": Hook("lift"), "cam_temporal": Hook("cam_temporal", pick=lambda o: o[0]),
                              "radar_in": Hook("radar.net", when="in"), "lift_all": Hook("lift", select="list")})
    rec.arm(["lift", "cam_temporal"])       # only these are recorded in the next forward pass(es)
    ... model(x) ...
    arrays = rec.read(batch_index=0)        # name -> numpy array of one sample (batch axis removed), float32
    rec.reset(); rec.close()

A Hook names the module by its dotted attribute path ('trunk.head', 'blocks.3'), whether the OUTPUT or the INPUT (first positional
argument) is taken, a `pick` to select from a tuple/dict output, and `select`: 'last' (the last call wins), 'first', or 'list'
(every call, for modules called once per frame / per sample; read() returns the list stacked on a new axis 0). Nothing is
recorded for taps that are not armed, so the hooks cost nothing in a normal forward. torch is imported only here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np


@dataclass
class Hook:
    path: str
    when: str = "out"                                  # 'out' | 'in'
    pick: Optional[Callable[[Any], Any]] = None
    select: str = "last"                               # 'last' | 'first' | 'list'


def resolve(model, path: str):
    """the module at a dotted path, or None when the network has no such part (optional blocks, other snapshots)"""
    obj = model
    for part in path.split("."):
        try:
            obj = obj[int(part)] if part.isdigit() else getattr(obj, part, None)
        except (IndexError, KeyError, TypeError):
            return None
        if obj is None:
            return None
    return obj


class TapRecorder:
    def __init__(self, model, hooks: Dict[str, Hook]):
        self.model, self.hooks = model, hooks
        self.armed: set = set()
        self.data: Dict[str, List[Any]] = {}
        self.errors: Dict[str, int] = {}
        self._handles = []
        for name, h in hooks.items():
            mod = resolve(model, h.path)
            if mod is None:                            # an optional module this network does not have
                continue
            self._handles.append(mod.register_forward_pre_hook(self._pre(name, h)) if h.when == "in"
                                 else mod.register_forward_hook(self._post(name, h)))

    def _store(self, name: str, h: Hook, value: Any) -> None:
        """record one value; a tap that cannot be read from this network's output is skipped (counted in `errors`), it never
        breaks the forward pass"""
        if name not in self.armed:
            return
        try:
            if h.pick is not None:
                value = h.pick(value)
            v = value.detach().float().cpu()
        except Exception:                                                           # noqa: BLE001
            self.errors[name] = self.errors.get(name, 0) + 1
            return
        lst = self.data.setdefault(name, [])
        if h.select == "last":
            lst[:] = [v]
        elif h.select == "first":
            if not lst:
                lst.append(v)
        else:
            lst.append(v)

    def _post(self, name, h):
        return lambda mod, args, out: self._store(name, h, out)

    def _pre(self, name, h):
        return lambda mod, args: self._store(name, h, args[0])

    def arm(self, names: Sequence[str]) -> None:
        unknown = [n for n in names if n not in self.hooks]
        if unknown:
            raise KeyError(f"unknown taps {unknown}")
        self.armed = set(names)
        self.data = {}

    def reset(self) -> None:
        self.data = {}

    def read(self, batch_index: Optional[int] = 0) -> Dict[str, np.ndarray]:
        """name -> numpy array; batch_index selects one sample along axis 0 of every recorded tensor (None keeps the batch)"""
        out = {}
        for name, lst in self.data.items():
            if not lst:
                continue
            arrs = [t[batch_index] if (batch_index is not None and t.ndim > 0) else t for t in lst]
            out[name] = (np.stack([a.numpy() for a in arrs]) if self.hooks[name].select == "list" else arrs[-1].numpy()).astype(np.float32)
        return out

    def close(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []
