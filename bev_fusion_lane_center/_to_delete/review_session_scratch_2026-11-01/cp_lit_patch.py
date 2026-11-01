p = "autoresearch/tools/commit_paths.py"
s = open(p, encoding="utf-8").read()
def sub(old, new):
    global s
    assert s.count(old) == 1, (old[:50], s.count(old))
    s = s.replace(old, new)
sub("def run(top, *args, env=None, inp=None, check=True):\n    env = dict(env if env is not None else os.environ, GIT_LITERAL_PATHSPECS=\"1\")      # names are literal: no globs, no `:(top)` magic",
    "def run(top, *args, env=None, inp=None, check=True, literal=True):\n    env = dict(env if env is not None else os.environ)\n    env.pop(\"GIT_LITERAL_PATHSPECS\", None)\n    if literal:\n        env[\"GIT_LITERAL_PATHSPECS\"] = \"1\"                                  # names are literal: no globs, no `:(top)` magic (check-ignore does not support it)")
sub('run(top, "check-ignore", "-q", "--no-index", "--", rel, check=False).returncode == 0', 'run(top, "check-ignore", "-q", "--no-index", "--", rel, check=False, literal=False).returncode == 0')
open(p, "w", encoding="utf-8", newline="\n").write(s)
print("ok")
