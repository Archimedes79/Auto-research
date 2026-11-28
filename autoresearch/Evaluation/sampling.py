"""Which scenes and frames to look at, seeded, so a subset can be reproduced: `scenes` draws whole scenes, `spread` takes frames evenly
spaced over the flat, time-ordered list of all frames (every scene in proportion to its length).
"""
from __future__ import annotations

from typing import List

import numpy as np

from .contracts import FrameRef, Source


class Sampler:
    def __init__(self, source: Source, seed: int = 0):
        self.source, self.seed = source, seed
        self.rng = np.random.default_rng(seed)
        self._frames = {s: source.frames(s) for s in source.scenes()}
        self._scenes = [s for s, fr in self._frames.items() if fr]
        if not self._scenes:
            raise ValueError("the source has no frames")

    def scenes(self, m: int) -> List[str]:
        """m different scenes (all of them when fewer exist), in random order"""
        idx = self.rng.permutation(len(self._scenes))[:m]
        return [self._scenes[i] for i in idx]

    def spread(self, n: int) -> List[FrameRef]:
        """n frames evenly spaced over the flat time-ordered list of the frames of all scenes"""
        flat = [fr for s in self._scenes for fr in self._frames[s]]
        if n >= len(flat):
            return flat
        idx = sorted({int(i) for i in np.linspace(0, len(flat) - 1, n).round()})
        return [flat[i] for i in idx]
