"""The contracts of the evaluation suite: three boxes you implement (Source, Subject, Task) and the data between them.

    Source   scenes ("videos") and frames, how to load one frame            -> Sample
    Subject  the network under test: reset(), predict(sample, taps)         -> Prediction
    Task     what the outputs mean: Metrics (additive counts), Views (pages of panels), plausibility checks

Everything the core needs to know about a frame is in `FrameRef`, `Sample`, `Prediction`; everything it draws is a `Page`
of `Panel`s (pure data, no plotting code in a View); everything it computes is a `Metric` with additive counts. The core
never imports a network, a dataset or a plotting library here (numpy only), so adapters of any size plug in.
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

Counts = Dict[str, np.ndarray]            # additive per-frame quantities of one Metric (fixed shape, or ragged along axis 0)


# ------------------------------------------------------------------------------------------------ identity
@dataclass(frozen=True, order=True)
class FrameRef:
    """One frame of one video. `scene` is unique in its Source, `index` is the position inside the scene (0 = the first frame
    the Source offers, time order), `key` is the dataset's own id (a token), optional."""
    scene: str
    index: int
    key: str = field(default="", compare=False)

    @property
    def label(self) -> str:
        return f"{self.scene}/{self.index:03d}"

    @property
    def slug(self) -> str:
        return f"{self.scene}_{self.index:03d}"

    def __str__(self) -> str:
        return self.label

    @staticmethod
    def parse(text: str) -> Optional[Tuple[str, int]]:
        """'scene-0061/12', 'scene-0061:12' or 'scene-0061/012' -> ('scene-0061', 12); None when it has no frame part."""
        m = re.match(r"^(.+?)[/:#@](\d+)$", text.strip())
        return (m.group(1), int(m.group(2))) if m else None


# ------------------------------------------------------------------------------------------------ data between the boxes
@dataclass
class Sample:
    """One loaded frame. `inputs` goes to the Subject untouched, `reference` (ground truth) to the Metrics and Views untouched,
    `display` holds extra things for pictures (an image, calibration, ...; values may be callables for lazy loading, read
    them with `get`), `meta` is free text-like information (timestamp, weather, ...)."""
    ref: FrameRef
    inputs: Any
    reference: Any = None
    display: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

    def get(self, name: str, default: Any = None) -> Any:
        v = self.display.get(name, default)
        if callable(v):
            v = v()
            self.display[name] = v
        return v


@dataclass
class Prediction:
    """What a Subject returned for one frame. `outputs` is in the Task's own terms (the Metrics and Views read it), `taps` are
    the intermediate arrays that were asked for (name -> numpy array), `info` is free (timings, loss values, ...)."""
    outputs: Any
    taps: Dict[str, np.ndarray] = field(default_factory=dict)
    info: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TapSpec:
    """A place inside the network that can be looked at. kind: 'feature_map' (C, H, W) | 'map' (H, W) | 'image' (H, W, 3 or 1)
    | 'logits' (K, H, W) | 'vector' (N,) | 'scalar'. `stage` groups taps in the picture (camera, radar, fusion, heads, ...),
    `note` says what the tap should look like, `redflag` what a problem looks like, `extent` places a map on a plane
    (left, right, bottom, top) and `flip_x` inverts its horizontal axis (e.g. +y on the left in a bird's-eye view). `optional`: the
    network may not have this part (another configuration, an older snapshot): not returning it is no contract violation."""
    name: str
    kind: str = "feature_map"
    stage: str = ""
    note: str = ""
    redflag: str = ""
    extent: Optional[Tuple[float, float, float, float]] = None
    flip_x: bool = False
    optional: bool = False
    alive_check: bool = True          # False: all-zero is legitimate (e.g. a memory on the first frame): no 'dead stage' flag


@dataclass
class Flag:
    """A plausibility finding. status 'ok' | 'warn' | 'fail'; heuristics flag, a human judges."""
    name: str
    status: str = "ok"
    value: Any = None
    note: str = ""


# ------------------------------------------------------------------------------------------------ pictures as data
@dataclass
class Shape:
    """An overlay on a panel, in the panel's own coordinates (pixels for 'image', plot coordinates for 'raster' / 'rgb').
    kind: 'polygon' (closed, xy (N, 2)) | 'line' (xy (N, 2)) | 'points' (xy (N, 2)) | 'circle' (xy = (cx, cy, r)) |
    'arrow' (xy = (x0, y0, x1, y1)) | 'text' (xy = (x, y), label is the text)."""
    kind: str
    xy: Any
    color: Any = "k"
    ls: str = "-"
    lw: float = 1.0
    label: str = ""


@dataclass
class Panel:
    """One picture. kind: 'image' (H, W[, 3] pixels), 'raster' (H, W scalar field, colour-mapped, optional colour bar), 'rgb'
    (H, W, 3 in 0..1, drawn at `extent`), 'text' (data = string), 'table' (data = list of rows, first row = header),
    'bars' (data = {name: value}). `extent` = (left, right, bottom, top) of a raster / rgb in plot coordinates with the first
    array axis upwards (origin lower); `xlim` / `ylim` override the plot limits (e.g. an inverted horizontal axis)."""
    kind: str
    title: str = ""
    data: Any = None
    extent: Optional[Tuple[float, float, float, float]] = None
    cmap: str = "magma"
    origin: str = "lower"            # 'lower': first array axis points up (maps on a plane); 'upper': image-plane arrays
    vmin: Optional[float] = None
    vmax: Optional[float] = None
    colorbar: bool = False
    xlim: Optional[Tuple[float, float]] = None
    ylim: Optional[Tuple[float, float]] = None
    equal: bool = True
    shapes: List[Shape] = field(default_factory=list)
    note: str = ""                    # one line for the html guide: what to look for


@dataclass
class Page:
    """A figure: panels in a grid (`ncols` columns), a title line, a colour legend and a footer line."""
    title: str
    panels: List[Panel]
    ncols: int = 3
    legend: List[Tuple[str, Any]] = field(default_factory=list)       # (label, colour)
    footer: str = ""
    panel_size: Tuple[float, float] = (4.6, 4.2)                      # inches per panel (width, height)


@dataclass
class ViewContext:
    """What a View gets besides the sample and the prediction."""
    scores: Dict[str, float] = field(default_factory=dict)            # per-frame scores of the metrics ('metric.key' -> value)
    taps: List[TapSpec] = field(default_factory=list)                 # the taps the Subject declares
    flags: List[Flag] = field(default_factory=list)
    subject: Dict[str, Any] = field(default_factory=dict)             # Subject.describe()
    wanted_taps: Optional[Sequence[str]] = None                       # the debug view: which taps to draw (None = all)


# ------------------------------------------------------------------------------------------------ the three boxes
class Source(ABC):
    """The dataset: scenes ("videos") and frames in time order."""

    @abstractmethod
    def scenes(self) -> List[str]:
        """names of all scenes, in a fixed order"""

    @abstractmethod
    def frames(self, scene: str) -> List[FrameRef]:
        """the frames of one scene in time order; `index` is the position in this list"""

    @abstractmethod
    def load(self, ref: FrameRef) -> Sample:
        """load one frame"""

    def describe(self) -> Dict[str, Any]:
        """free information for the manifest (dataset, split, version, ...)"""
        return {"class": type(self).__name__}

    def prefetch(self, refs: Sequence[FrameRef]) -> None:
        """optional hint: these frames will be loaded next, in this order"""

    # -- identification (default implementation on top of scenes / frames)
    def find(self, spec: Any) -> FrameRef:
        """A FrameRef from 'scene-0061/12', 'scene-0061:12', ('scene-0061', 12), a FrameRef or a dataset key (token)."""
        if isinstance(spec, FrameRef):
            return self.frames(spec.scene)[spec.index]
        if isinstance(spec, (tuple, list)) and len(spec) == 2:
            scene, idx = spec[0], int(spec[1])
        else:
            text = str(spec)
            parsed = FrameRef.parse(text)
            if parsed is None:
                for s in self.scenes():                      # a dataset key (token)
                    for r in self.frames(s):
                        if r.key and r.key == text:
                            return r
                raise KeyError(f"{text!r}: not 'scene/index' and not a frame key; scenes start with {self.scenes()[:3]}")
            scene, idx = parsed
        if scene not in self.scenes():
            import difflib
            raise KeyError(f"unknown scene {scene!r}; closest: {difflib.get_close_matches(scene, self.scenes(), 3)}")
        fr = self.frames(scene)
        if not 0 <= idx < len(fr):
            raise KeyError(f"{scene} has frames 0..{len(fr) - 1}, asked {idx}")
        return fr[idx]


class Subject(ABC):
    """The network under test. Stateful along a scene: the Runner calls reset() at the start of a scene (or before every frame
    in state='reset' mode) and then predict() frame by frame in time order."""
    warmup: int = 0                  # frames of history the network needs before a frame is meaningful (memory warm-up)

    @abstractmethod
    def predict(self, sample: Sample, taps: Sequence[str] = ()) -> Prediction:
        """run the network on one frame; return the tapped arrays named in `taps` (numpy, one sample, no batch axis)"""

    def reset(self) -> None:
        """forget the state (start of a scene)"""

    def taps(self) -> List[TapSpec]:
        """the places that can be looked at"""
        return []

    def describe(self) -> Dict[str, Any]:
        return {"class": type(self).__name__}

    def benchmark(self, sample: Sample, n: int = 30, warmup: int = 5) -> Dict[str, float]:
        """ms per call of predict() on one frame (median, p90, mean); override to time only the network and to synchronise
        a device. Valid only on an otherwise idle machine."""
        import time
        for _ in range(warmup):
            self.predict(sample)
        t = []
        for _ in range(n):
            t0 = time.perf_counter()
            self.predict(sample)
            t.append((time.perf_counter() - t0) * 1000.0)
        t = np.sort(np.asarray(t))
        return {"median_ms": float(np.median(t)), "p90_ms": float(t[int(0.9 * (len(t) - 1))]), "mean_ms": float(t.mean()),
                "n": n}

    def close(self) -> None:
        """release the network"""


class Metric(ABC):
    """One family of numbers, in a form that can be traced: additive counts per frame, a reduction of any set of frames to
    numbers, per-frame scalars for ranking outliers.

    frame(): Counts for ONE frame (dict name -> numpy array; arrays must be additive or concatenable: `reduce` receives the
        list of Counts of any set of frames).
    reduce(): the numbers of that set, a flat dict 'name' -> float (nan allowed); with `keys` only those numbers are needed
        (a metric may use that to skip expensive ones: the bootstrap calls it thousands of times). The dataset-level number is
        reduce(all), never a mean of per-frame numbers.
    score(): per-frame scalars (e.g. the frame's own IoU) to rank frames; support(): how much evidence each score rests on
        (e.g. the GT cell count), so a frame with nothing to find is not ranked as an outlier.
    headline: the reduce() keys that tables, outlier search and bootstrap use; better: per key 'high' | 'low' (default high)."""
    name: str = "metric"
    headline: Tuple[str, ...] = ()
    better: Dict[str, str] = {}

    @abstractmethod
    def frame(self, sample: Sample, pred: Prediction) -> Counts:
        """additive counts of one frame"""

    @abstractmethod
    def reduce(self, counts: Sequence[Counts], keys: Optional[Sequence[str]] = None) -> Dict[str, float]:
        """numbers of a set of frames (all keys, or at least `keys`)"""

    def score(self, counts: Counts) -> Dict[str, float]:
        return {}

    def support(self, counts: Counts) -> Dict[str, float]:
        return {}

    def direction(self, key: str) -> str:
        return self.better.get(key, "high")


class View(ABC):
    """A way to draw one frame: sample + prediction -> Page."""
    name: str = "overview"
    description: str = ""

    @abstractmethod
    def render(self, sample: Sample, pred: Prediction, ctx: ViewContext) -> Page:
        """the page of this frame"""


class Task(ABC):
    """What the outputs of a Subject mean: its Metrics, its Views, its plausibility checks."""
    name: str = "task"
    default_video_view: str = "overview"

    @abstractmethod
    def metrics(self) -> List[Metric]:
        """all metrics (unique names)"""

    @abstractmethod
    def views(self) -> Dict[str, View]:
        """name -> View; must contain 'overview' (ground truth next to the prediction with the errors highlighted)"""

    def checks(self, sample: Sample, pred: Prediction) -> List[Flag]:
        return []

    def guide(self) -> str:
        """text for the html index: how to read the pictures"""
        return ""

    def probe(self, sample: Sample, pred: Prediction) -> Dict[str, float]:
        """optional, for the probing (`probing.py`): numbers of ONE frame, stored with it. 'tag.<name>': a property of the situation
        (0 / 1, or a number: curvature, range, night ...); 'cause.<name>': the amount of error assigned to a cause, in one unit that
        adds up over frames (e.g. metres of lane). Cheap, deterministic, no network access."""
        return {}

    def causes(self) -> Dict[str, Tuple[str, str]]:
        """optional: cause name (without the 'cause.' prefix) -> (what it is, what to improve: model / code / evaluation)"""
        return {}
