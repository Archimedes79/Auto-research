"""python -m runsuite Runner.predict runs/x small.val -- every method of the classes `Runner` (run a network: predict, plan, list_results, tap, time, probe) and `RunJobs` (the runner as jobs:
check, run, status, queue_add and the queue methods of Training) is a command; the short forms `predict runs/x small.val` and `queue-add job.json` work too.
`python -m runsuite methods` lists them. The queue is the one of the trainings, so `python -m trainsuite queue-list` shows runner jobs as well. Arguments and output: autoresearch/suitecli.py."""
import sys

from runsuite import cli


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv and argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    rc = cli.dispatch(argv)
    if rc is None:
        print(f"runsuite: {'no command' if not argv else 'unknown command ' + repr(argv[0])}; `python -m runsuite methods` lists them\n\n{__doc__}", file=sys.stderr)
        return 2
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
