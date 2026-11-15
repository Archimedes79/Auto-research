"""trainsuite - watch, stop and judge a training run from the files it writes (standard library; curves need matplotlib).

    python -m trainsuite status                       # table of the runs under ./runs
    python -m trainsuite health t8_bevfeat_s0 --vs t8_ctl_s0 --kill "group.lines@1000 > 1.48"
    python -m trainsuite watch t8_bevfeat_s0 t8_ctl_s0 --kill "group.lines@1000 > 1.48" --enforce
    python -m trainsuite curves t8_bevfeat_s0 t8_ctl_s0
    python -m trainsuite stop t8_bevfeat_s0

The suite never imports project code and never touches a training process: it reads the run directory (contract: README.md) and, to
stop, writes a CONTROL file that the trainer reads. The project's model, losses and data stay in the project.

The interface of the training system is the class Training of api.py (its head and the README, section 0); every method is also a command (`python -m trainsuite methods`).
"""
from .control import EXIT_STOPPED, clear_stale, honours_stop, mark_stopped, request_stop, stop_requested  # noqa: F401
from .health import Finding  # noqa: F401   (health.check(run) is the health check of a run; the package-level `check` is the one of a job)
from .kill import Rule, load_rules  # noqa: F401
from .runs import Run, read_log  # noqa: F401
from .api import JOB_FIELDS, Training, TrainingError, load_job  # noqa: F401,E402
