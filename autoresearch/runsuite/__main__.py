"""python -m runsuite Runner.run job.json -- every method of the class Runner is a command (`Runner.check`, `Runner.run`, `Runner.status`, `Runner.queue_add`, and the queue methods of
Training: `Runner.queue_list`, ...); without the class name it works too (`run job.json`). `python -m runsuite methods` lists them. The queue is the one of the trainings, so
`python -m trainsuite queue ...` shows runner jobs as well. Arguments and output: see trainsuite/apicli.py."""
import sys

import runsuite
from trainsuite import apicli


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    rc = apicli.dispatch(runsuite.Runner, argv, "runsuite")
    if rc is None:
        print(__doc__, file=sys.stderr)
        return 2
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
