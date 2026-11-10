"""autogt - the old name of GtGenerator/ (renamed 2026-11-10, D76), kept ONLY for the pinned code snapshots of runs made before the rename.

A stored run keeps a copy of its code (runs/<name>/code/*.py) that says `from autogt.nuscenes_adapter import pose_matrix`, finds the repository by the folder
`autogt/` and, when it is held, restarts from that copy: it must keep working. So `import autogt` is the package GtGenerator itself and `import autogt.X` is the
module GtGenerator.X - the same module objects, no second copy of any state (a test of `python -m GtGenerator selftest` pins that).

New code writes GtGenerator. `python -m autogt COMMAND` works, `python -m autogt.X` does not (use `python -m GtGenerator.X`).
Remove this folder and its three lines in .gitignore when no stored run is needed any more.
"""
import importlib
import importlib.abc
import importlib.util
import sys


class _Alias(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """autogt.X -> the module object GtGenerator.X (imported when asked for; its spec is left as it was)"""

    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith("autogt.") and not fullname.endswith(".__main__"):          # __main__: the package's own file runs (python -m autogt)
            return importlib.util.spec_from_loader(fullname, self)
        return None

    def create_module(self, spec):
        module = importlib.import_module("GtGenerator" + spec.name[len("autogt"):])
        spec.original = module.__spec__                  # the import machinery overwrites __spec__ of the module it is given; exec_module puts it back
        return module

    def exec_module(self, module):
        module.__spec__ = module.__spec__.original


if not any(type(finder).__name__ == "_Alias" for finder in sys.meta_path):
    sys.meta_path.insert(0, _Alias())
sys.modules[__name__] = importlib.import_module("GtGenerator")                           # `import autogt` hands out the package GtGenerator
