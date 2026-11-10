"""runner - a trained network run on frames: `from runner import Runner` (the class, runner.py) and `python -m runner` (the command line, __main__.py). bev_fusion must be on the path (the working
folder, or PYTHONPATH). bev_runner.py (the planner's runner) is a script and not imported from here."""
from runner.runner import Runner, RunnerError  # noqa: F401
