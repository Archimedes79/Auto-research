"""runsuite - run a trained network on frames and produce the data for the evaluation, with the same interface as the training (trainsuite) and the same queue.

    from runsuite import RunJobs
    r = RunJobs("../bev_fusion")                        # the project folder (default: AR_PROJECT)
    job = {"id": "run.t20.val8", "kind": "run", "mode": "full", "run": "t20_base_s0", "split": "val", "n": 8, "dts": [0, 0.5]}     # or the path of a json file
    r.check(job)         # are the fields and paths right, which command would run                      -> {ok, problems, command, out}
    r.run(job)           # start it now (run.py: record, ledger row "no card", GPU slot)                   -> {id, started, out, command}
    r.queue_add(job)     # put it in the SAME queue as the trainings (kind "run")                           -> the item
    r.status("run.t20.val8")   # state, exit code, duration, the files it wrote, the tail of its log        -> dict

The queue methods of Training (queue_list, queue_get, queue_remove, queue_start, queue_hold, queue_resume) are the same here: one queue for trainings and runner jobs. Every method is
also a command: `python -m runsuite run job.json` (the method name with hyphens: `queue-add`, `queue-list`). The project names its runner in project.json: "runsuite": {"command": "...{args}...", "runs": "runs"}; {args} is where the mode
and the flags go: `<mode> <target> [--split S] [--n N | --scene X [--first N]] [--out DIR] [--dts a b] [--device D] [--config C] [--cold] [--code DIR]` (bev_fusion/runner/bev_runner.py).
The modes: check (builds the input, writes nothing), precompute (layer 1 alone), network (layers 2-5 on a precompute folder), full (both, in one process, frame by frame; it writes
objects.json and maps.npz, the data the evaluation reads). The convention is autoresearch/INTERFACE.md. The same methods are also functions of a default object: `runsuite.check(job)`.
"""
from trainsuite.api import TrainingError as RunJobsError  # noqa: F401
from .api import (RunJobs, check, queue_add, queue_get, queue_hold, queue_list, queue_remove, queue_resume, queue_start, run, status)  # noqa: F401
