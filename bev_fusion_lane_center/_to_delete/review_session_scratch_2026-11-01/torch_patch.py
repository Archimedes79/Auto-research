"""one-off: the manifest reads python / torch / cuda / numpy from the interpreter the run was started with"""
p = "autoresearch/tools/provenance.py"
s = open(p, encoding="utf-8").read()


def sub(old, new):
    global s
    assert s.count(old) == 1, (old[:60], s.count(old))
    s = s.replace(old, new)


sub('''    block["environment"] = {"python": r.get("python") or UNKNOWN, "gpu": r.get("gpu") or UNKNOWN, "host": r.get("host") or UNKNOWN,
                            "conda_env": r.get("conda_env") or UNKNOWN, "torch": UNKNOWN, "source": "run record"}
''', '''    env = trainer_env(r)
    block["environment"] = {"python": env.get("python") or r.get("python") or UNKNOWN, "torch": env.get("torch") or UNKNOWN, "cuda": env.get("cuda") or UNKNOWN,
                            "numpy": env.get("numpy") or UNKNOWN, "gpu": r.get("gpu") or UNKNOWN, "host": r.get("host") or UNKNOWN,
                            "conda_env": r.get("conda_env") or UNKNOWN, "source": env.get("source") or "run record (the trainer's interpreter was not found)"}
''')
sub('''def manifest(run_id, run_dir=None, write=True, force=False):''', '''PROBE = ("import json, platform\\nfrom importlib import metadata\\n"
         "def v(n):\\n    try:\\n        return metadata.version(n)\\n    except Exception:\\n        return None\\n"
         "print(json.dumps({'python': platform.python_version(), 'torch': v('torch'), 'numpy': v('numpy')}))\\n")


def trainer_python(r):
    """the interpreter a run was started with: project.json "trainer_python", else the .../bin/python that the script named in the run's command calls
    (bev_fusion: temp/train/run_ar.sh -> ~/miniconda3/envs/gcar/bin/python); None when it is not found on this machine"""
    cands = [P.cfg_of(P.root()).get("trainer_python")]
    cwd = r.get("cwd") or P.root()
    for tok in (r.get("command") or "").split():
        f = os.path.join(cwd, tok)
        if tok.endswith(".sh") and os.path.isfile(f):
            try:
                cands += re.findall(r"(~?/[\\w./-]*/bin/python[\\d.]*)", open(f, encoding="utf-8", errors="replace").read())
            except OSError:
                pass
    for c in cands:
        c = os.path.expanduser(c) if c else c
        if c and os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    return None


def trainer_env(r):
    """python, torch, CUDA and numpy versions of the interpreter that ran the training: asked through the package metadata (torch is not imported:
    about 0.1 s); the CUDA version is the +cuNNN part of the torch version; {} when the interpreter is not found or does not answer"""
    py = trainer_python(r)
    if not py:
        return {}
    try:
        out = subprocess.run([py, "-c", PROBE], capture_output=True, text=True, timeout=30)
        d = json.loads(out.stdout.strip().splitlines()[-1])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return {}
    m = re.search(r"\\+cu(\\d+)", d.get("torch") or "")
    d["cuda"] = f"{m.group(1)[:-1]}.{m.group(1)[-1]}" if m else None
    d["source"] = f"the interpreter of the run: {py}"
    return d


def manifest(run_id, run_dir=None, write=True, force=False):''')
# selftest: a fake interpreter, found through the script named in the run's command
sub('''        m = manifest("r2")
''', '''        fake = os.path.join(tmp, "envs", "bin")
        os.makedirs(fake)
        open(os.path.join(fake, "python"), "w").write("#!/bin/sh\\necho '{\\"python\\": \\"3.10.12\\", \\"torch\\": \\"2.8.0+cu128\\", \\"numpy\\": \\"1.26.4\\"}'\\n")
        os.chmod(os.path.join(fake, "python"), 0o755)
        open(os.path.join(tmp, "train.sh"), "w").write("#!/bin/bash\\n" + os.path.join(fake, "python") + " -u train.py\\n")
        rj2 = os.path.join(proj, "rr", "r2", "run.json")
        d2 = json.load(open(rj2))
        d2.update(command="bash train.sh", cwd=tmp)
        json.dump(d2, open(rj2, "w"))
        m = manifest("r2")
        assert m["environment"]["torch"] == "2.8.0+cu128" and m["environment"]["cuda"] == "12.8" and m["environment"]["numpy"] == "1.26.4" and "envs/bin/python" in m["environment"]["source"], m["environment"]
        assert trainer_env({"command": "bash nothing.sh", "cwd": tmp}) == {} and trainer_env({}) == {}, "an interpreter that is not found reads unknown, not a guess"
''')
open(p, "w", encoding="utf-8", newline="\n").write(s)
print("ok")
