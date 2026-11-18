p = "evaluation/jprune_b.py"
s = open(p, encoding="utf-8").read()
s = s.replace("import argparse\nimport copy\nimport hashlib", "import argparse\nimport contextlib\nimport copy\nimport hashlib")
s = s.replace('''def load_models(seed, device):''', '''@contextlib.contextmanager
def cpu_solve_fix(device):
    """B's section stage (`sections.py`) solves a batched 512 x 512 linear system with `torch.linalg.solve`. On this WSL box the threaded oneMKL LU fails for it ("Parameter 6 was incorrect on entry to SLASWP") and takes minutes per batch; with one thread it takes 0.03 s and
    gives the residual of float32 (6e-7). On the CPU the solve therefore runs single-threaded for the duration of the pass (the same numbers up to the rounding of the other algorithm; no digested file is edited); on the GPU nothing changes"""
    if str(device) != "cpu":
        yield
        return
    orig = torch.linalg.solve

    def solve1(A, B, **kw):
        n = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            return orig(A, B, **kw)
        finally:
            torch.set_num_threads(n)
    torch.linalg.solve = solve1
    try:
        yield
    finally:
        torch.linalg.solve = orig


def load_models(seed, device):''')
s = s.replace('''    frames, preds, nb, t0 = [], [], 0, time.time()
    with torch.no_grad():''', '''    frames, preds, nb, t0 = [], [], 0, time.time()
    with torch.no_grad(), cpu_solve_fix(device):''')
open(p, "w", encoding="utf-8").write(s)
