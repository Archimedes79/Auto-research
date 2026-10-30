"""The Runner: frames through the Subject, in the order and with the state a deployed system would see.

    iter_scene(scene)      every frame of a scene in time order; state='carry' keeps the Subject's memory from frame to frame
                           (reset() once at the start), state='reset' makes every frame independent (reset() before each)
    run_to(ref)            random access: reset, feed `Subject.warmup` history frames (their outputs are discarded), then the
                           frame itself (taps on). cold=True skips the history.
The Runner owns nothing but the order of calls; metrics, pictures and storage are the callers' business.
"""
from __future__ import annotations

import queue
import threading
import time
from typing import Iterator, List, Optional, Sequence, Tuple

from .contracts import FrameRef, Prediction, Sample, Source, Subject


class Runner:
    def __init__(self, source: Source, subject: Subject, state: str = "carry", prefetch: int = 2):
        if state not in ("carry", "reset"):
            raise ValueError("state must be 'carry' or 'reset'")
        self.source, self.subject, self.state, self.prefetch = source, subject, state, prefetch

    # -------------------------------------------------------------------------------------- iteration
    def _loader(self, refs: Sequence[FrameRef]) -> Iterator[Sample]:
        """load the frames in order; a worker thread reads ahead (disk and decoding overlap the network)"""
        if self.prefetch <= 0 or len(refs) < 3:
            for r in refs:
                yield self.source.load(r)
            return
        q: "queue.Queue" = queue.Queue(maxsize=self.prefetch)
        stop = threading.Event()
        self.source.prefetch(refs)

        def work():
            try:
                for r in refs:
                    if stop.is_set():
                        return
                    q.put(self.source.load(r))
                q.put(None)
            except BaseException as e:                                        # noqa: BLE001
                q.put(e)
        t = threading.Thread(target=work, daemon=True)
        t.start()
        try:
            while True:
                item = q.get()
                if item is None:
                    return
                if isinstance(item, BaseException):
                    raise item
                yield item
        finally:
            stop.set()
            try:
                while True:
                    q.get_nowait()
            except queue.Empty:
                pass

    def iter_scene(self, scene: str, frames: Optional[Sequence[FrameRef]] = None, taps: Sequence[str] = ()
                   ) -> Iterator[Tuple[Sample, Prediction]]:
        """(sample, prediction) of the frames of one scene in time order (all frames, or the given ones, which must be in
        time order). In 'carry' mode skipping frames skips their effect on the state: pass consecutive frames."""
        refs = list(frames) if frames is not None else self.source.frames(scene)
        if self.state == "carry":
            self.subject.reset()
        for sample in self._loader(refs):
            if self.state == "reset":
                self.subject.reset()
            yield sample, self._predict(sample, taps)

    def _predict(self, sample: Sample, taps: Sequence[str]) -> Prediction:
        t0 = time.perf_counter()
        pred = self.subject.predict(sample, taps)
        pred.info.setdefault("predict_ms", (time.perf_counter() - t0) * 1000.0)
        return pred

    def run_to(self, ref: FrameRef, taps: Sequence[str] = (), cold: bool = False, history=None) -> Tuple[Sample, Prediction]:
        """one frame with the history a deployed system would have in the same scene. history: None = Subject.warmup frames
        before it (fast; exact only if the network forgets older frames), 'full' = every frame from the start of the scene (the
        exact replay of a carried run: what a trace of a quantitative run needs), an int = that many, 'none' / cold = none.
        In state 'reset' there is never a history."""
        scene_frames = self.source.frames(ref.scene)
        if cold or self.state == "reset" or history in ("none", 0):
            n_hist = 0
        elif history == "full":
            n_hist = ref.index
        elif history is None:
            n_hist = min(int(self.subject.warmup), ref.index)
        else:
            n_hist = min(int(history), ref.index)
        self.subject.reset()
        for r in scene_frames[ref.index - n_hist:ref.index]:
            self.subject.predict(self.source.load(r), ())
        sample = self.source.load(scene_frames[ref.index])
        return sample, self._predict(sample, taps)
