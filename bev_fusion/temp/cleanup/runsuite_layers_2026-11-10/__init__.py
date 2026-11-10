"""runsuite - run a trained network on frames. Three layers, each calls only the one below:

    cli.py     the command line: `python -m runsuite Runner.predict runs/x small.val --taps fused`, every method of a class is a command (suitecli.py)
    api.py     the class interface: `Runner(project)` with predict, plan, list_results, tap, time, probe; no queue, no launcher, a call runs in this process
    engine.py  the functionality: `Engine`, the run without ground truth on the pieces of evalsuite (ResultSet, Frames, the frame iterator); nothing else knows how a network runs
    jobs.py    beside them: `RunJobs`, the same and the planner's runner of the project as JOBS (run.py: ledger row, GPU slot; the queue of trainsuite) - a job of mode "predict" starts Runner.predict

    from runsuite import Runner
    r = Runner("../bev_fusion")                                      # the project folder (default: AR_PROJECT)
    r.plan("runs/t23_C_s1", "small.val")                             # the dry run of predict: what is stored, what would run, the problems          -> dict
    r.predict("runs/t23_C_s1", "small.val", taps=["fused"])          # the network on the frames, no ground truth, stored (warm, whole scenes: D70)   -> {results: "<net>@r<hash>", ...}
    r.list_results("runs/t23_C_s1");  r.tap(id, "scene-0061/12", "fused");  r.time("runs/t23_C_s1");  r.probe("runs/t23_C_s1")

    from runsuite import RunJobs                                     # the job layer: imports trainsuite, so only when asked for
    j = RunJobs("../bev_fusion");  j.check("pred.t23.small.json");  j.run(job);  j.queue_add(job);  j.status("pred.t23.small")

Details and the job file: README.md. The convention: autoresearch/README.md, 'The interface of the suites'."""
from .api import Runner, RunnerError  # noqa: F401


def __getattr__(name):                                          # the job layer and the command line load trainsuite: not for whoever only runs a network
    if name in ("RunJobs", "RunJobsError", "CLI", "dispatch"):
        from . import cli, jobs
        return {"RunJobs": jobs.RunJobs, "RunJobsError": jobs.T.TrainingError, "CLI": jobs.CLI, "dispatch": cli.dispatch}[name]
    raise AttributeError(f"module 'runsuite' has no attribute {name!r}")
