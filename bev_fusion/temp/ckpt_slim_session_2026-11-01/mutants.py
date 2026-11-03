"""does the selftest of training/scratch.py fail when one of the fixes is taken out? (run from anywhere; works on copies)"""
import os
import shutil
import subprocess
import sys
import tempfile

SRC = sys.argv[1]
text = open(SRC).read()
MUTANTS = [
    ("superseded loop deletes the file it just wrote", "        if p == path:  ", "        if False:  "),
    ("stale .done marker kept", '        os.remove(p + ".done")', "        pass"),
    ("missing dependency ignored", "        if not hits:\n", "        if False:\n"),
    ("guard on a missing folder", "shutil.disk_usage(_existing(path or ROOT))", "shutil.disk_usage(path or ROOT)"),
    ("empty sidecar crashes", "    except (OSError, ValueError):", "    except OSError:"),
    ("unreadable cache crashes", "        except (OSError, EOFError, RuntimeError, pickle.UnpicklingError):", "        except OSError:"),
    ("non-list build accepted", "    if not isinstance(obj, list):\n        raise TypeError(f\"build(", "    if False:\n        raise TypeError(f\"build("),
    ("drive check removed", "    if free - gb < keep:", "    if False:"),
]
killed = 0
for name, old, new in MUTANTS:
    assert text.count(old) == 1, (name, text.count(old))
    d = tempfile.mkdtemp()
    os.makedirs(os.path.join(d, "training"))
    open(os.path.join(d, "training", "scratch.py"), "w").write(text.replace(old, new))
    r = subprocess.run([sys.executable, os.path.join(d, "training", "scratch.py"), "--selftest"], capture_output=True, text=True,
                       env=dict(os.environ, OMP_NUM_THREADS="2", CUDA_VISIBLE_DEVICES=""))
    dead = r.returncode != 0
    killed += dead
    print(("KILLED   " if dead else "SURVIVED ") + name)
    shutil.rmtree(d, ignore_errors=True)
print(f"{killed} of {len(MUTANTS)} mutants killed")
