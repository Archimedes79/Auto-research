"""Which scenes and frames to look at. Seeded, so a sample can be reproduced; the draws are the ones Mirko asked for:
first a video (scene) uniformly, then a frame uniformly inside it (so a frame of a short scene is more likely than a frame of
a long one - every video gets the same chance to be looked at). `spread` is the other kind of subset: evenly spaced over the
flat, time-ordered list of all frames (what a quantitative subset needs: every scene in proportion to its length).
"""
from __future__ import annotations

from typing import List, Optional, Sequence

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

    @property
    def n_frames(self) -> int:
        return sum(len(f) for f in self._frames.values())

    def frames(self, n: int, unique: bool = True, scenes: Optional[Sequence[str]] = None) -> List[FrameRef]:
        """n draws of (scene uniform, frame uniform in the scene); unique: a frame drawn twice is drawn again."""
        pool = [s for s in (scenes if scenes is not None else self._scenes) if self._frames.get(s)]
        total = sum(len(self._frames[s]) for s in pool)
        if unique and n > total:
            raise ValueError(f"asked {n} unique frames, only {total} exist")
        out: List[FrameRef] = []
        seen = set()
        guard = 0
        while len(out) < n:
            guard += 1
            if guard > 1000 * max(n, 1):
                raise RuntimeError("could not draw enough distinct frames")
            s = pool[int(self.rng.integers(0, len(pool)))]
            fr = self._frames[s][int(self.rng.integers(0, len(self._frames[s])))]
            if unique and fr.label in seen:
                continue
            seen.add(fr.label)
            out.append(fr)
        return out

    def scenes(self, m: int, min_frames: int = 1) -> List[str]:
        """m different scenes (all of them when fewer exist), in random order"""
        pool = [s for s in self._scenes if len(self._frames[s]) >= min_frames]
        idx = self.rng.permutation(len(pool))[:m]
        return [pool[i] for i in idx]

    def spread(self, n: int, scenes: Optional[Sequence[str]] = None) -> List[FrameRef]:
        """n frames evenly spaced over the flat time-ordered list of the frames of `scenes` (default all scenes)."""
        flat = [fr for s in (scenes if scenes is not None else self._scenes) for fr in self._frames.get(s, [])]
        if n >= len(flat):
            return flat
        idx = sorted({int(i) for i in np.linspace(0, len(flat) - 1, n).round()})
        return [flat[i] for i in idx]
