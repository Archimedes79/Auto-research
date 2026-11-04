"""scratch: which trainkit does the train.py of a run snapshot import? usage: check_pinned.py <run folder>   (sys.path[0] = the snapshot folder, as when train.py runs as a script)"""
import os
import sys

code = os.path.abspath(os.path.join(sys.argv[1], "code"))
sys.path.insert(0, code)
import train  # noqa: E402
import trainkit  # noqa: E402
print("snapshot", os.path.relpath(code, os.getcwd()), "| trainkit imported from", os.path.relpath(trainkit.__file__, os.getcwd()), "| loop from", os.path.relpath(train.fit.__code__.co_filename, os.getcwd()))
