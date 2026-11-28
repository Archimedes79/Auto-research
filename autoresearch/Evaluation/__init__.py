"""Evaluation - what a network does, as numbers and pictures, on one path: predict -> stored predictions -> score -> read.

    from Evaluation import Evaluation              # the class interface of a project (api.py); command line: python -m Evaluation methods
A project's adapter implements the boxes of contracts.py (Source, Subject, Task with its Metrics and Views). README.md: use; EXTENSIONS.md: the adapter contract and the rest."""
from .contracts import (Counts, Flag, FrameRef, Metric, Page, Panel, Prediction, Sample, Shape, Source, Subject, TapSpec, Task,
                        View, ViewContext)
from .suite import EvalSuite
from .api import Evaluation, EvaluationError, Frames

__all__ = ["EvalSuite", "Source", "Subject", "Task", "Metric", "View", "FrameRef", "Sample", "Prediction", "Page", "Panel", "Shape",
           "TapSpec", "Flag", "ViewContext", "Counts", "Evaluation", "Frames", "EvaluationError"]
