"""Two helpers that several model gates had a copy of: `flat` (the outputs of a model call as one flat dict of tensors) and `refuser` (does `model.check_config` refuse a config).
    from model_gates import flat, refuser          # tests/support and training/ on sys.path
    refused = refuser(TINY)                        # refused(cam_buffer="stack") -> True when check_config(ModelConfig(**dict(TINY, cam_buffer="stack"))) raises ValueError"""
import torch

from model import ModelConfig, check_config


def flat(res):
    """{(dt, key, subkey): tensor} of res["out"] (a nested dict dt -> key -> tensor or dict of tensors); subkey is "" for a tensor directly under the key"""
    d = {}
    for dt, p in res["out"].items():
        for k, v in p.items():
            for kk, t in (v.items() if isinstance(v, dict) else [("", v)]):
                if torch.is_tensor(t):
                    d[(dt, k, kk)] = t
    return d


def refuser(tiny):
    """refused(**kw): True when check_config raises ValueError for the model config `tiny` with the fields kw changed"""
    def refused(**kw):
        try:
            check_config(ModelConfig(**dict(tiny, **kw)))
            return False
        except ValueError:
            return True
    return refused
