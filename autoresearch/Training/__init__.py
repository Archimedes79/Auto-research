"""Training - watch, stop and judge a training run from the files it writes (standard library; curves need matplotlib).

    python -m Training list                         # the trainings of the last day and the box
    python -m Training health t8_bevfeat_s0 --vs t8_ctl_s0 --kill "group.lines@1000 > 1.48"
    python -m Training watch t8_bevfeat_s0 t8_ctl_s0 --kill "group.lines@1000 > 1.48" --enforce
    python -m Training plot t8_bevfeat_s0
    python -m Training stop t8_bevfeat_s0

The suite never imports project code and never touches a training process: it reads the run directory (contract: README.md) and, to
stop, writes a CONTROL file that the trainer reads. The project's model, losses and data stay in the project.

The interface of the training system is the class Training of api.py (its head and the README, section 0); every method is also a command (`python -m Training methods`).
"""
from .api import Training, TrainingError  # noqa: F401
