"""scratch launcher: run a script in a process where the old package name `autogt` points at GtGenerator (runs pinned before D76 import it; the alias was removed on purpose, D78 / D81:
'put an alias into the code that runs it'). Usage: python temp/t26_b/with_autogt_alias.py SCRIPT.py args ...   Nothing is copied: autogt.X is the module GtGenerator.X."""
import importlib
import importlib.abc
import importlib.util
import os
import runpy
import sys

WS = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(0, WS)


class OldName(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, name, path=None, target=None):
        if name == "autogt" or name.startswith("autogt."):
            return importlib.util.spec_from_loader(name, self)

    def create_module(self, spec):
        return importlib.import_module("GtGenerator" + spec.name[len("autogt"):])

    def exec_module(self, module):
        pass


sys.meta_path.insert(0, OldName())
script = os.path.abspath(sys.argv[1])
sys.argv = [script] + sys.argv[2:]
sys.path.insert(0, os.path.dirname(script))
runpy.run_path(script, run_name="__main__")
