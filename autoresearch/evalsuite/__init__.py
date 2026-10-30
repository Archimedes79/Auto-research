"""evalsuite - a qualitative and a quantitative evaluation suite for any network, behind three small interface boxes.

    from evalsuite import EvalSuite, Source, Subject, Task          # implement the three for your network (adapters/toy.py shows how)
    suite = EvalSuite(source, subject, task, out="eval_out")
    suite.qual.frame("scene-0061/12", views=("overview", "debug"));  suite.qual.random(6);  suite.qual.videos(2)
    run = suite.quant.run(n_scenes=20);  suite.quant.report(run)
Command line: python -m evalsuite --adapter pkg.module:build [--arg key=value] <qual|quant|selftest|selftest-adapter> ...
See README.md (use) and ARCHITECTURE.md (design, how to port).
"""
from .contracts import (Counts, Flag, FrameRef, Metric, Page, Panel, Prediction, Sample, Shape, Source, Subject, TapSpec, Task,
                        View, ViewContext)
from .suite import EvalSuite

__all__ = ["EvalSuite", "Source", "Subject", "Task", "Metric", "View", "FrameRef", "Sample", "Prediction", "Page", "Panel", "Shape",
           "TapSpec", "Flag", "ViewContext", "Counts"]
__version__ = "1.0"
