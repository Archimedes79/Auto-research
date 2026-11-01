"""one-off: provenance.py repro <id> --run: check the run's tag out in a throw-away worktree under temp/repro/ and print the command that repeats the run"""
p = "autoresearch/tools/provenance.py"
s = open(p, encoding="utf-8").read()


def sub(old, new):
    global s
    assert s.count(old) == 1, (old[:70], s.count(old))
    s = s.replace(old, new)


sub('''    d["source"] = f"the interpreter of the run: {py}"''', '''    d["source"] = f"the interpreter of the run, read {dt.date.today().isoformat()} (the environment may have changed since the run): {py}"''')
sub('''def find_run_dir(run_id):''', '''def repro_prepare(run_id, dest=None):
    """Make a run repeatable without touching the current checkout: the run's code tag is checked out (detached) in a worktree under temp/repro/<id>
    (or `dest`), data/ and <problem>/data/ are linked to the real ones, and the command that repeats the run is printed, not started: it goes through
    run.py as `<id>.repro` with --no-card (a repetition feeds no verdict) and writes into the worktree's own runs/ folder, so the original run stays as
    it is. Remove it later with `git worktree remove --force <dir>`. -> (worktree dir, command line)"""
    rj = os.path.join(P.path("run_records"), run_id, "run.json")
    if not os.path.isfile(rj):
        sys.exit(f"no run record {rj}")
    r = json.load(open(rj, encoding="utf-8"))
    tag, repo = r.get("tag"), repo_root()
    if not tag or git(repo, "rev-parse", "--verify", "--quiet", f"refs/tags/{tag}") is None:
        sys.exit(f"the code tag {tag} of {run_id} does not exist here (git fetch --tags): nothing to check out")
    dest = os.path.abspath(dest or os.path.join(repo, "temp", "repro", run_id))
    if not os.path.isdir(dest):
        if git(repo, "worktree", "add", "--detach", dest, tag) is None:
            sys.exit(f"git worktree add {dest} {tag} failed")
    for rel in ("data", os.path.join(os.path.relpath(P.root(), repo), "data")):
        src, dst = os.path.join(repo, rel), os.path.join(dest, rel)
        if os.path.isdir(src) and not os.path.lexists(dst):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            os.symlink(src, dst)
    cwd = r.get("cwd") or P.root()
    rel = os.path.relpath(cwd, repo) if os.path.commonpath([os.path.abspath(cwd), repo]) == repo else os.path.relpath(P.root(), repo)
    work = os.path.normpath(os.path.join(dest, rel))
    cmd = (f"cd {work} && AR_PROJECT=. python3 ../autoresearch/tools/run.py --id {run_id}.repro --hyp \\"repetition of {run_id}\\" "
           f"--pred \\"same code, same GT, same seed: the numbers agree within seed noise\\" --rule \\"by hand: a repetition, no verdict\\" --no-card -- {r.get('command')}")
    return dest, cmd


def find_run_dir(run_id):''')
sub('''    ap.add_argument("--force", action="store_true", help="manifest: rebuild an existing provenance.json")''', '''    ap.add_argument("--force", action="store_true", help="manifest: rebuild an existing provenance.json")
    ap.add_argument("--run", action="store_true", help="repro: check the tag out in a worktree and print the repeat command")
    ap.add_argument("--dir", help="repro --run: where to put the worktree (default temp/repro/<id>)")''')
sub('''        repro(a.args[0], a.card)
''', '''        repro(a.args[0], a.card)
        if a.run:
            wt, cmd = repro_prepare(a.args[0], a.dir)
            print(f"\\nworktree (the tag checked out, the current checkout untouched): {wt}\\nstart the repetition (check the GT store above first):\\n  {cmd}\\nremove it afterwards: git worktree remove --force {wt}")
''')
# selftest: a real worktree of a tag in a throw-away repository
sub('''        # the model bundle (network + losses = the import closure of model.py and losses.py) and the manifest of a training run''', '''        # repro --run: the tag is checked out in a worktree, data is linked, the current checkout is untouched
        run("tag", "tg/r1")
        real_root = repo_root
        globals()["repo_root"] = lambda start=None: tmp
        try:
            os.makedirs(os.path.join(tmp, "data", "autogt"), exist_ok=True)
            wt, cmd = repro_prepare("r1")
            assert os.path.isfile(os.path.join(wt, "ev", "m.py")) and wt == os.path.join(tmp, "temp", "repro", "r1"), wt
            assert os.path.islink(os.path.join(wt, "data")) and os.path.realpath(os.path.join(wt, "data")) == os.path.realpath(os.path.join(tmp, "data")), "data is linked"
            assert "--id r1.repro" in cmd and "--no-card" in cmd and "bash e.sh" in cmd and cmd.startswith("cd "), cmd
            assert repro_prepare("r1")[0] == wt, "a second call reuses the worktree"
            assert not os.path.exists(os.path.join(tmp, "temp", "repro", "r1", "ckpt.pt")) and os.path.isfile(os.path.join(tmp, "ev", "m.py")), "the checkout is separate"
            try:
                repro_prepare("r2")
                raise AssertionError("a run without its tag cannot be prepared")
            except SystemExit as e:
                assert "does not exist" in str(e) or "no run record" in str(e), e
        finally:
            globals()["repo_root"] = real_root
        # the model bundle (network + losses = the import closure of model.py and losses.py) and the manifest of a training run''')
open(p, "w", encoding="utf-8", newline="\n").write(s)
print("ok")
