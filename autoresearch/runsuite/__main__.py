# COMMENTED OUT 2026-11-10 (Mirko: "Auskommentieren, nicht loeschen"): the runner is not a runsuite. It is ONE class `Runner` in bev_fusion/runner/ with `python -m runner` above it,
# parallel to Training and Evaluation. This job layer (a tracked start through run.py and a queue item of kind run around a runner command) stays here as comments, in git history
# at commit 75da8962 and dd30e4ac, and can be taken back when a job layer is wanted.
raise SystemExit("runsuite is commented out (Mirko 2026-11-10): the runner is the class Runner in bev_fusion/runner/, run it with: python -m runner")
# """python -m runsuite RunJobs.run job.json -- every method of the class RunJobs is a command (`RunJobs.check`, `RunJobs.run`, `RunJobs.status`, `RunJobs.queue_add`, and the queue methods of
# Training); the short forms `run job.json` and `queue-add job.json` work too. `python -m runsuite methods` lists them. The queue is the one of the trainings, so `python -m trainsuite queue-list`
# shows runner jobs as well. Arguments and output: autoresearch/suitecli.py."""
# import sys
#
# import runsuite
#
#
# def main(argv=None):
#     argv = sys.argv[1:] if argv is None else list(argv)
#     if argv and argv[0] in ("-h", "--help"):
#         print(__doc__)
#         return 0
#     rc = runsuite.CLI.dispatch(argv)
#     if rc is None:
#         print(f"runsuite: {'no command' if not argv else 'unknown command ' + repr(argv[0])}; `python -m runsuite methods` lists them\n\n{__doc__}", file=sys.stderr)
#         return 2
#     return rc
#
#
# if __name__ == "__main__":
#     raise SystemExit(main())
