"""tools/run_ar.sh, the launcher that trains one config under autoresearch/tools/run.py (D73: it was a scratch script until then, and the pinned engine rested on one line of it that was
not committed). A throw-away git repository (training/ with a toy trainer, a toy autoresearch/trainkit), CPU, a few seconds, no GPU:
  1. the snapshot: training/ and autoresearch/trainkit of the commit are unpacked into runs/<name>/code/, SNAPSHOT.txt names the commit, the log gets the header line, the trainer
     runs from the copy and imports ITS trainkit (the pin), the exit code of the trainer is the exit code of the launcher, `.done` exists only for 0
  2. a run trains the commit it was given (AR_COMMIT, else the second argument, else HEAD), never the working tree, with the config copy of that commit; an unknown commit stops
     the launch; a config outside training/ is used as given
  3. a commit without autoresearch/trainkit fails the launch (git archive), no trainer runs, no `.done`
  4. the old path temp/train/run_ar.sh (about 60 run records and the held runs name it) is executed: same commit, run id, exit code
  5. project.json and the launch line of startable.py name tools/run_ar.sh
Run from the bev_fusion folder: wsl ~/miniconda3/envs/gcar/bin/python tests/test_run_ar.py
"""
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(os.path.dirname(HERE), "tools", "run_ar.sh")
TRAINER = ('import json, os, sys\nimport trainkit\nprint("trainkit from", os.path.realpath(trainkit.__file__))\nprint("version", "%s")\n'
           'sys.exit(json.load(open(sys.argv[1])).get("exit", 0))\n')


def git(repo, *args):
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@example.com", "-C", repo, *args], capture_output=True, text=True, check=True).stdout.strip()


def launch(repo, cfg, env_extra=None, args=(), script="tools/run_ar.sh"):
    env = dict(os.environ, AR_RUN_ID="t.run", **(env_extra or {}))
    if not (env_extra or {}).get("AR_COMMIT"):
        env.pop("AR_COMMIT", None)
    r = subprocess.run(["bash", script, cfg, *args], cwd=os.path.join(repo, "bev_fusion"), env=env, capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


def main():
    tmp = tempfile.mkdtemp(prefix="run_ar_")
    try:
        repo = os.path.join(tmp, "repo")
        for d in ("bev_fusion/training", "bev_fusion/tools", "autoresearch/trainkit"):
            os.makedirs(os.path.join(repo, d))
        shutil.copy(SCRIPT, os.path.join(repo, "bev_fusion", "tools", "run_ar.sh"))
        open(os.path.join(repo, "bev_fusion", "training", "train.py"), "w").write(TRAINER % "V1")
        open(os.path.join(repo, "bev_fusion", "training", "ok.json"), "w").write('{"exit": 0}')
        open(os.path.join(repo, "bev_fusion", "training", "bad.json"), "w").write('{"exit": 3}')
        open(os.path.join(repo, "autoresearch", "trainkit", "__init__.py"), "w").write("")
        git(repo, "init", "-q")
        git(repo, "add", "-A")
        git(repo, "commit", "-q", "-m", "one")
        c1 = git(repo, "rev-parse", "HEAD")
        run = os.path.join(repo, "bev_fusion", "runs")

        # 1 the snapshot and the pin, the exit code, the done flag, the tail that run.py reads
        rc, out = launch(repo, "training/ok.json")
        log = open(os.path.join(run, "ok", "train.log")).read()
        assert rc == 0 and os.path.exists(os.path.join(run, "ok", ".done")), (rc, out[-300:])
        assert os.path.exists(os.path.join(run, "ok", "code", "train.py")) and os.path.exists(os.path.join(run, "ok", "code", "trainkit", "__init__.py")), "training/ and trainkit are unpacked"
        assert c1[:10] in open(os.path.join(run, "ok", "code", "SNAPSHOT.txt")).read() and f"commit {c1[:10]}" in log and "run t.run" in log, "the header line names the commit and the run"
        assert os.path.realpath(os.path.join(run, "ok", "code", "trainkit", "__init__.py")) in log, f"the trainer imports the trainkit next to it (the pin): {log[-300:]}"
        assert "version V1" in out and "ok exit 0" in out, f"the last lines of train.log and the exit line are printed (run.py's CUDA retry and last_lines read them): {out[-300:]}"
        rc, out = launch(repo, "training/bad.json")
        assert rc == 3 and not os.path.exists(os.path.join(run, "bad", ".done")), "the exit code of the trainer is the exit code of the launcher, .done only for 0"
        print("  ok   snapshot of training/ and trainkit, header line, the trainer imports its pinned trainkit, exit code, .done, the printed tail")

        # 2 a run trains the commit it was given, not HEAD and not the working tree: HEAD moves on (V2, a config that exits 5), the working tree has an edit (V3)
        open(os.path.join(repo, "bev_fusion", "training", "train.py"), "w").write(TRAINER % "V2")
        open(os.path.join(repo, "bev_fusion", "training", "ok.json"), "w").write('{"exit": 5}')
        git(repo, "add", "-A")
        git(repo, "commit", "-q", "-m", "two")
        c2 = git(repo, "rev-parse", "HEAD")
        open(os.path.join(repo, "bev_fusion", "training", "train.py"), "w").write(TRAINER % "V3")

        def last(name):
            return open(os.path.join(run, name, "train.log")).read().split("=== ")[-1]

        rc, out = launch(repo, "training/ok.json", {"AR_COMMIT": c1}, args=(c2,))
        assert rc == 0 and "version V1" in last("ok") and f"commit {c1[:10]}" in last("ok"), "AR_COMMIT (the commit run.py made) wins over the second argument and over HEAD"
        assert "config runs/ok/code/ok.json" in last("ok"), "the config is the copy inside the snapshot (the commit's config, not the one of HEAD that exits 5)"
        rc, out = launch(repo, "training/ok.json", args=(c1,))
        assert rc == 0 and "version V1" in last("ok") and f"commit {c1[:10]}" in last("ok"), "without AR_COMMIT the second argument is the commit (HEAD is c2 here)"
        rc, out = launch(repo, "training/ok.json")
        assert rc == 5 and "version V2" in last("ok"), "without both HEAD, and never the working tree (V3)"
        rc, out = launch(repo, "training/ok.json", {"AR_COMMIT": "0" * 40})
        assert rc == 2 and "no commit" in out, "an unknown commit stops the launch before anything runs"
        outside = os.path.join(tmp, "outside.json")                       # a config outside training/ is used as given
        shutil.copy(os.path.join(repo, "bev_fusion", "training", "ok.json"), outside)
        rc, out = launch(repo, outside, {"AR_COMMIT": c1})
        assert rc == 5 and f"config {outside}" in last("outside"), (rc, out[-300:])
        print("  ok   a run trains the commit it was given (not HEAD, not the working tree) with that commit's config copy; a config outside training/ is used as given")

        # 3 a commit without the engine fails the launch
        shutil.rmtree(os.path.join(repo, "autoresearch"))
        git(repo, "add", "-A")
        git(repo, "commit", "-q", "-m", "three, without trainkit")
        shutil.rmtree(os.path.join(run, "ok"))
        rc, out = launch(repo, "training/ok.json")
        assert rc != 0 and not os.path.exists(os.path.join(run, "ok", ".done")) and not os.path.exists(os.path.join(run, "ok", "train.log")), (rc, out[-300:])
        print("  ok   a commit without autoresearch/trainkit fails the launch: no trainer, no .done")

        # 4 the old path (held runs and about 60 run records name it) is executed: it hands over to the launcher with the environment, the arguments and the exit code
        import re
        old_file = os.path.join(os.path.dirname(HERE), "temp", "train", "run_ar.sh")
        os.makedirs(os.path.join(repo, "bev_fusion", "temp", "train"))
        shutil.copy(old_file, os.path.join(repo, "bev_fusion", "temp", "train", "run_ar.sh"))
        rc, out = launch(repo, "training/bad.json", {"AR_COMMIT": c1}, script="temp/train/run_ar.sh")
        assert rc == 3 and "version V1" in last("bad") and f"commit {c1[:10]}" in last("bad") and "run t.run" in last("bad") and not os.path.exists(os.path.join(run, "bad", ".done")), (rc, out[-300:])
        code = "\n".join(ln for ln in open(old_file, encoding="utf-8").read().splitlines() if not ln.lstrip().startswith("#"))
        assert re.findall(r"(~?/[\w./-]*/bin/python[\d.]*)", code.replace("$HOME", "~")), "the redirect still names the interpreter (provenance.trainer_python reads it from the script a run's command names)"
        print("  ok   the old path temp/train/run_ar.sh runs the launcher: commit, run id, exit code; it still names the interpreter")

        # 5 the project's entries name the launcher that exists: project.json (train, fingerprint) and the launch line that startable prints
        import json
        proj = json.load(open(os.path.join(os.path.dirname(HERE), "project.json"), encoding="utf-8"))
        assert proj["trainsuite"]["train"] == "bash tools/run_ar.sh {config}" and proj["fingerprint_extra"] == ["tools/run_ar.sh"], "project.json names tools/run_ar.sh"
        assert "tools" in proj["structure"]["root_dirs"] and os.path.isfile(SCRIPT)
        sys.path.insert(0, os.path.join(os.path.dirname(HERE), "training"))
        import startable
        line = startable.start_lines(os.path.join(os.path.dirname(HERE), "training", "configs", "x.json"), "t99_x_s0")
        assert "-- bash tools/run_ar.sh training/configs/x.json" in line, line
        print("  ok   project.json and the printed launch line name tools/run_ar.sh")
        print("RUN_AR LAUNCHER: PASS")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
