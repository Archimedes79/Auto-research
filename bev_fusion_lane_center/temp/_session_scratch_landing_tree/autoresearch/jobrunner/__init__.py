"""jobrunner - run a method of a suite as a job under run.py, with its data named in the job. README.md has the fields; `python -m jobrunner methods` lists the commands.

    from jobrunner import JobRunner
    JobRunner(project).submit({"id": "t23.C_s1.eval", "call": "Evaluation.run", "args": {"network": "t23.C_s1", "frames": "large.val"}})"""
from .api import JobError, JobRunner  # noqa: F401
