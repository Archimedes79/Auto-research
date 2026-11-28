"""Pictures of one frame: a view of the Task, or the generic look into the network (`debug`), drawn from a loaded frame and its prediction. `Evaluation.draw_frame` draws the stored
prediction of a frame with it; no network runs here."""
from __future__ import annotations

from typing import Dict, Optional, Sequence

from .contracts import Page, Prediction, Sample, TapSpec, Task, View, ViewContext
from .render import figure, taps as tapmod


class DebugView(View):
    """the generic look into the network: one panel per tap (by kind) and a table of their statistics with the plausibility flags"""
    name = "debug"
    needs_taps = True

    def render(self, sample: Sample, pred: Prediction, ctx: ViewContext) -> Page:
        panels = tapmod.tap_panels(pred.taps, ctx.taps)
        panels.append(tapmod.stats_table(pred.taps, ctx.flags))
        return Page(title="debug: network parts", panels=panels, ncols=4, panel_size=(3.6, 3.4))


def views_of(task: Task) -> Dict[str, View]:
    """the Task's views, and `debug` when it has none of that name"""
    return {"debug": DebugView(), **task.views()}


def draw(task: Task, name: str, sample: Sample, pred: Prediction, path: str, specs: Sequence[TapSpec] = (), subject: Optional[Dict] = None) -> Dict[str, float]:
    """render the view `name` of one frame into the png `path`, headed by the frame's label and scores; -> the scores (every Metric's score() of this frame, as the counting stores them).
    A view that looks inside the network (needs_taps) also gets the TapSpecs and the description of the subject, and the plausibility flags of the Task and of the taps"""
    view = views_of(task)[name]
    scores: Dict[str, float] = {}
    for m in task.metrics():
        scores.update({f"{m.name}.{k}": float(v) for k, v in m.score(m.frame(sample, pred)).items()})
    flags = []
    if getattr(view, "needs_taps", False):
        flags = list(task.checks(sample, pred)) + [flag for spec in specs if spec.name in pred.taps for flag in tapmod.plausibility(spec, pred.taps[spec.name])]
    page = view.render(sample, pred, ViewContext(scores=scores, taps=list(specs), flags=flags, subject=subject or {}))
    head = "   ".join([f"{sample.ref.label}  [{name}]"] + [f"{k} {v:.3f}" for k, v in scores.items() if v == v][:8])
    page.title = f"{head}\n{page.title}" if page.title else head
    figure.render_page(page, path)
    return scores
