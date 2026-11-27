"""The checks of the evaluation (run by `python -m Evaluation.selftest`): every module registers its checks with `@check("what it proves")`.
This folder is not part of the code hash of a network's identity (defaults.CODE_SKIP): a change of a test predicts nothing again."""
from typing import Callable, List, Tuple

CHECKS: List[Tuple[str, Callable]] = []


def check(name: str):
    def register(fn):
        CHECKS.append((name, fn))
        return fn
    return register


def raises(exc, text: str, fn, *args, **kwargs) -> None:
    """fn(*args) raises `exc` with `text` in its message"""
    try:
        fn(*args, **kwargs)
    except exc as e:
        assert text in str(e), (text, str(e))
        return
    raise AssertionError(f"no {exc.__name__} with {text!r}")
