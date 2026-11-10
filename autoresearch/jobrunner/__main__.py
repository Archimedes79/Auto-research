import sys

import suitecli

from . import call
from .api import JobError, JobRunner

CLI = suitecli.Cli(JobRunner, "jobrunner", JobError, "JOBRUNNER_TRACEBACK", positional=("job", "i"))


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv[:1] == ["call"]:
        return call.main(argv[1:])
    if argv[:1] == ["selftest"]:
        from . import selftest
        return selftest.main()
    rc = CLI.dispatch(argv)
    if rc is None:
        print("usage: python -m jobrunner methods | JobRunner.<method> ... | call ... | selftest", file=sys.stderr)
        return 2
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
