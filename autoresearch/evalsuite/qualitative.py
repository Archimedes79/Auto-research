"""The qualitative suite: pictures of one frame, of n random frames, of the inside of the network, and videos of sequences.

    frame(spec, views, taps, cold)      one concrete frame ('scene-0061/12', ('scene-0061', 12), a frame key); history frames
                                        are fed first when the network needs them (Subject.warmup), cold=True skips them
    debug(spec, taps)                   the same frame with the network parts: tap pictures, statistics, plausibility flags,
                                        and the raw taps as <slug>_taps.npz
    random(n, seed, views)              n draws, each: a random video first, then a random frame in it (seeded, no repeats)
    videos(m, seed, view, fps)          m random sequences run in order with the state carried, one picture per frame -> mp4,
                                        plus a strip of the per-frame scores so the failing moment is visible
Every picture carries the frame label and the frame's own scores in its header, so it can be found again and compared with
the numbers of the quantitative suite (the scores are the Metrics' `score()` of this very frame).
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .contracts import Flag, Page, Prediction, Sample, View, ViewContext
from .render import figure, report, taps as tapmod, video
from .runner import Runner
from .sampling import Sampler


@dataclass
class QualResult:
    ref: str
    images: Dict[str, str] = field(default_factory=dict)
    taps_npz: str = ""
    scores: Dict[str, float] = field(default_factory=dict)
    flags: List[Dict[str, Any]] = field(default_factory=list)


class DebugView(View):
    """the generic look into the network: one panel per tap (by kind), a statistics table, the plausibility flags"""
    name = "debug"
    description = "network parts: tap pictures, statistics and plausibility flags"
    needs_taps = True

    def render(self, sample: Sample, pred: Prediction, ctx: ViewContext) -> Page:
        panels = tapmod.tap_panels(pred.taps, ctx.taps, ctx.wanted_taps)
        panels.append(tapmod.stats_table(pred.taps, ctx.flags))
        return Page(title="debug: network parts", panels=panels, ncols=4, panel_size=(3.6, 3.4))


def score_line(scores: Dict[str, float], limit: int = 8) -> str:
    items = [f"{k} {v:.3f}" for k, v in scores.items() if v == v][:limit]
    return "   ".join(items)


class Qualitative:
    def __init__(self, suite):
        self.s = suite
        self.dir = os.path.join(suite.out, "qualitative")

    # -------------------------------------------------------------------------------------- helpers
    def _views(self) -> Dict[str, View]:
        v = dict(self.s.task.views())
        v.setdefault("debug", DebugView())
        if "overview" not in v:
            raise KeyError("the Task must offer an 'overview' view")
        return v

    def _need_taps(self, views: Sequence[str], taps: Optional[Sequence[str]]) -> List[str]:
        declared = [t.name for t in self.s.subject.taps()]
        if taps is not None:
            unknown = [t for t in taps if t not in declared]
            if unknown:
                raise KeyError(f"unknown taps {unknown}; declared: {declared}")
            return list(taps)
        vs = self._views()
        return declared if any(getattr(vs[v], "needs_taps", False) for v in views) else []

    def context(self, sample: Sample, pred: Prediction, wanted: Optional[Sequence[str]] = None) -> ViewContext:
        """per-frame scores of every Metric, the Task's flags and the tap flags for this frame"""
        scores: Dict[str, float] = {}
        for m in self.s.task.metrics():
            sc = m.score(m.frame(sample, pred))
            scores.update({f"{m.name}.{k}": float(v) for k, v in sc.items()})
        specs = {t.name: t for t in self.s.subject.taps()}
        flags: List[Flag] = list(self.s.task.checks(sample, pred))
        for name, arr in pred.taps.items():
            if name in specs:
                flags += tapmod.plausibility(specs[name], arr)
        return ViewContext(scores=scores, taps=list(specs.values()), flags=flags, subject=self.s.subject.describe(),
                           wanted_taps=wanted)

    def page(self, sample: Sample, pred: Prediction, view: str, ctx: ViewContext, header: str = "") -> Page:
        p = self._views()[view].render(sample, pred, ctx)
        head = f"{sample.ref.label}  [{view}]"
        if ctx.scores:
            head += "   " + score_line(ctx.scores)
        p.title = f"{head}\n{p.title}" if p.title else head
        if header:
            p.title += f"\n{header}"
        return p

    def _draw(self, sample, pred, views, ctx, out_dir, header="") -> QualResult:
        res = QualResult(ref=sample.ref.label, scores=ctx.scores, flags=[asdict(f) for f in ctx.flags if f.status != "ok"])
        for v in views:
            path = os.path.join(out_dir, f"{sample.ref.slug}_{v}.png")
            figure.render_page(self.page(sample, pred, v, ctx, header), path)
            res.images[v] = path
        if pred.taps:
            res.taps_npz = os.path.join(out_dir, f"{sample.ref.slug}_taps.npz")
            np.savez_compressed(res.taps_npz, **{k.replace("/", "_"): np.asarray(v) for k, v in pred.taps.items()})
        return res

    # -------------------------------------------------------------------------------------- modes
    def frame(self, spec: Any, views: Sequence[str] = ("overview",), taps: Optional[Sequence[str]] = None, cold: bool = False,
              out_dir: Optional[str] = None, history=None, state: Optional[str] = None) -> QualResult:
        """history: None = Subject.warmup frames before the frame, 'full' = replay the scene from its start (exactly what a
        carried quantitative run saw), 'none' / cold = start empty. state: 'carry' | 'reset' (default: the suite's)."""
        ref = self.s.source.find(spec)
        names = self._need_taps(views, taps)
        sample, pred = Runner(self.s.source, self.s.subject, state or self.s.state).run_to(ref, names, cold=cold, history=history)
        ctx = self.context(sample, pred, taps)
        return self._draw(sample, pred, views, ctx, out_dir or os.path.join(self.dir, "frames"))

    def debug(self, spec: Any, taps: Optional[Sequence[str]] = None, cold: bool = False, history=None) -> QualResult:
        """the overview next to the debug view of one frame (taps: names to draw, default all declared)"""
        return self.frame(spec, views=("overview", "debug"), taps=taps, cold=cold, history=history)

    def random(self, n: int, seed: int = 0, views: Sequence[str] = ("overview",), taps: Optional[Sequence[str]] = None,
               scenes: Optional[Sequence[str]] = None) -> List[QualResult]:
        refs = Sampler(self.s.source, seed).frames(n, unique=True, scenes=scenes)
        out_dir = os.path.join(self.dir, f"random_s{seed}_n{n}")
        results = [self.frame(r, views, taps, out_dir=out_dir) for r in refs]
        sec = []
        for r in results:
            cell = "".join(report.image(p, out_dir, 560, f"{r.ref} [{v}]") for v, p in r.images.items())
            flags = "; ".join(f"{f['status']}: {f['name']}" for f in r.flags[:4])
            cmd = f"python -m evalsuite ... qual frame {r.ref}"
            sec.append(("", f"<p><b>{r.ref}</b> &nbsp; <span class=note>{score_line(r.scores)} {flags}</span><br>{cell}<br>"
                            f"<code>{cmd}</code></p>"))
        report.write(os.path.join(out_dir, "index.html"), f"random frames (seed {seed}, n {n})",
                     [("", "<p class=note>%s</p>" % self.s.task.guide().replace("\n", "<br>"))] + sec)
        with open(os.path.join(out_dir, "draws.json"), "w") as f:
            json.dump({"seed": seed, "n": n, "frames": [r.ref for r in results], "views": list(views)}, f, indent=1)
        return results

    def videos(self, m: int = 1, seed: int = 0, view: Optional[str] = None, fps: float = 4.0, max_frames: Optional[int] = None,
               scenes: Optional[Sequence[str]] = None, render_every: int = 1) -> List[Dict[str, Any]]:
        """m random scenes (or the named ones) as videos, one picture per frame, in the suite's state (carried along the scene by default)."""
        view = view or self.s.task.default_video_view
        names = list(scenes) if scenes else Sampler(self.s.source, seed).scenes(m)
        need = self._need_taps([view], None)
        out = []
        for scene in names:
            frames = self.s.source.frames(scene)[: max_frames or None]
            runner = Runner(self.s.source, self.s.subject, self.s.state)
            imgs, rows = [], []
            for k, (sample, pred) in enumerate(runner.iter_scene(scene, frames, need)):
                ctx = self.context(sample, pred)
                rows.append({"index": sample.ref.index, **ctx.scores})
                if k % max(1, render_every) == 0:
                    imgs.append(figure.render_page(self.page(sample, pred, view, ctx, f"frame {k + 1}/{len(frames)}")))
            h, w = imgs[0].shape[:2]
            imgs = [_fit(i, h, w) for i in imgs]
            vdir = os.path.join(self.dir, "videos")
            info = video.encode(imgs, os.path.join(vdir, f"{scene}_{view}.mp4"), fps=fps)
            strip = self._strip(scene, rows, os.path.join(vdir, f"{scene}_{view}_scores.png"))
            with open(os.path.join(vdir, f"{scene}_{view}_scores.json"), "w") as f:
                json.dump(rows, f, indent=0)
            out.append({"scene": scene, "view": view, "video": info, "scores_strip": strip, "frames": len(frames)})
        report.write(os.path.join(self.dir, "videos", "index.html"), f"videos (seed {seed}, view {view})",
                     [(o["scene"], f"<p><a href='{os.path.basename(o['video']['path'])}'>{os.path.basename(o['video']['path'])}</a>"
                                   f" ({o['video']['codec']}, {o['frames']} frames)</p>"
                                   f"{report.image(o['scores_strip'], os.path.join(self.dir, 'videos'), 900)}") for o in out])
        return out

    def _strip(self, scene: str, rows: List[Dict[str, float]], path: str) -> str:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        keys = [k for k in rows[0] if k != "index"][:6] if rows else []
        fig, ax = plt.subplots(figsize=(11, 3))
        for k in keys:
            ax.plot([r["index"] for r in rows], [r.get(k, np.nan) for r in rows], marker=".", lw=1, label=k)
        ax.set_xlabel("frame index")
        ax.set_title(f"{scene}: per-frame scores")
        ax.legend(fontsize=6, ncol=3)
        fig.tight_layout()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, dpi=80)
        plt.close(fig)
        return path


def _fit(img: np.ndarray, h: int, w: int) -> np.ndarray:
    if img.shape[:2] == (h, w):
        return img
    from PIL import Image
    return np.asarray(Image.fromarray(img).resize((w, h)))
