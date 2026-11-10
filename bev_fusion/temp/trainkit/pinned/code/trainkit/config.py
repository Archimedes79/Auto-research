"""The config of the engine: the fields `fit` reads. A project extends it (`@dataclass class TrainConfig(Config)`) with its own fields (model, data, loss options);
the whole config is written to `<run>/config.json` and into every checkpoint, and a resume compares it (checkpoint.check_resume)."""
import json
from dataclasses import dataclass, field


@dataclass
class Config:
    name: str = "run"
    out_root: str = "runs"                       # runs/<name>/ (derived data, not in git)
    accum: int = 8                               # micro-batches per optimiser step: effective batch = batch size * accum
    steps: int = 20000                           # OPTIMISER steps
    lr: float = 3e-4                             # published 2e-4 .. 4e-4 (BEVDet family, not scaled with the batch); fresh
                                                 # trunk / heads with BatchNorm: <= 6e-4 (lane project's derail note)
    lr_min_frac: float = 1e-3
    warmup: int = 500                            # linear from lr / 3 (scan 2026-10-26)
    weight_decay: float = 1e-2
    grad_clip: float = 5.0
    grad_clip_split: tuple = ()                  # module prefixes (as freeze / train_only) whose gradients are clipped to grad_clip ON THEIR OWN, the rest as one group: a block whose
                                                 # gradient is far above the others' (the motion block in t16: 36 - 131 against 0.1 - 17) no longer sets the clip factor of the whole
                                                 # network; () = one global clip as before (bit-identical)
    lr_groups: dict = field(default_factory=dict)  # module prefix -> {"lr": float, "warmup": int} (either key optional; the chain's lr / warmup otherwise): that module's parameters get their own
                                                 # learning rate and warm-up (optim.lr_group_params / param_groups); {} = today's two groups and one schedule, bit for bit. Not free on a resume
                                                 # (the optimiser state has a slot per group)
    ema: float = 0.999
    amp: str = "off"                           # "off" | "bf16" | "fp16" (CUDA only). fp16 trains with a GradScaler (dynamic loss scaling). bf16 as it is: ledger row speed.c2 (an 8-step diagnostic, with cudnn.benchmark): no speed-up, gradient norm 25x fp32
    fp16_scale: float = 1024.0                   # amp "fp16": the start scale of the GradScaler (halved on an overflow; a power of two scales exactly)
    fp16_growth_interval: int = 2000             # amp "fp16": the scale doubles after that many clean steps in a row (an overflow restarts the count); the torch default
    log_every: int = 20
    eval_every: int = 1000
    save_every: int = 500
    keep: int = 3
    slim_at_end: bool = False                    # True: a run that finishes keeps ONE checkpoint without optimiser state and RNG (about half
                                                 # the size, no older ones): it evaluates and warm-starts but cannot resume training.
                                                 # False (default; Mirko 2026-11-01: keep the full data to resume later): `keep` complete ones
    max_bad_steps: int = 20                     # consecutive non-finite steps before aborting
    seed: int = 0
    init_from: str = ""                          # checkpoint of another run to start from (checkpoint.warm_start; only while this run has
                                                 # no checkpoint of its own; optimiser, step and schedule start fresh), "" = off
    init_reset: tuple = ()                       # key prefixes that keep their fresh initialisation in that warm start (t11: a dead
                                                 # branch, e.g. "obj_head.head.vel.")
    device: str = "cuda"

    # not fields (no annotation): the rules of a resume, read by checkpoint.check_resume; a project's config class overrides them
    RESUME_FREE = ("name", "out_root", "device", "log_every", "eval_every", "save_every", "keep", "slim_at_end", "max_bad_steps")   # fields a resume may change (paths, the cadence of logging and saving); every other difference is refused
    RESUME_SWITCHES = ()                         # fields that a checkpoint from before them lacks and that change the numbers when turned on in the middle of a run
    RESUME_ENV = "TRAINKIT_RESUME_ANY"           # the environment variable that overrides the refusal (set to 1)


def read_config(cls, path):
    """`cls` (a Config or a subclass) from a json file; keys starting with "_" are comments for the reader (e.g. "_about"), every other unknown key is an error"""
    with open(path) as fh:
        return cls(**{k: v for k, v in json.load(fh).items() if not k.startswith("_")})
