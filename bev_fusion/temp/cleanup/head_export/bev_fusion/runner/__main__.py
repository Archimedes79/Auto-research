"""python -m runner Runner.predict runs/t23_C_s1 small.val --taps fused -- the command line of the class Runner (runner.py): every method is a command (the generator is autoresearch/suitecli.py,
the same for Training and Evaluation), `python -m runner methods` lists them. Run from the bev_fusion folder. No logic here: a new method of Runner is a new command without a line in this file."""
import sys

from runner.runner import Runner, RunnerError  # noqa: E402   (it puts autoresearch on the path)

import suitecli  # noqa: E402

CLI = suitecli.Cli(Runner, "runner", RunnerError, "RUNNER_TRACEBACK", positional=("network", "frames", "results", "frame", "name"))


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv and argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    rc = CLI.dispatch(argv)
    if rc is None:
        print(f"runner: {'no command' if not argv else 'unknown command ' + repr(argv[0])}; `python -m runner methods` lists them\n\n{__doc__}", file=sys.stderr)
        return 2
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
