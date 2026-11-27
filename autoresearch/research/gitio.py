"""git, the few ways the research system uses it: a snapshot commit of the working tree through a temporary index (the shared index and the
branch stay as they are) with an annotated tag; a commit of named paths onto the branch (what other chats edit stays out); reading a file
of a commit; unpacking a commit into a folder."""
from __future__ import annotations

import io
import os
import re
import subprocess
import tarfile
import tempfile
from typing import Any, Dict, List, Optional, Pattern, Sequence, Tuple

from research.paths import ResearchError


def git(root: str, *args: str, env: Optional[Dict[str, str]] = None, check: bool = True, data: Optional[bytes] = None) -> str:
    r = subprocess.run(["git", "-C", root, *args], capture_output=True, input=data, env=env)
    if check and r.returncode != 0:
        raise ResearchError(f"git {' '.join(args[:3])}: {r.stderr.decode('utf-8', 'replace').strip()[:300]}")
    return r.stdout.decode("utf-8", "replace").strip()


def head(root: str) -> str:
    return git(root, "rev-parse", "HEAD")


def branch(root: str) -> str:
    name = git(root, "symbolic-ref", "-q", "--short", "HEAD", check=False)
    if not name:
        raise ResearchError("HEAD is detached: check out a branch first")
    return name


def tag_commit(root: str, tag: str) -> str:
    """the commit a tag points at, '' when there is no such tag (a failure of git itself is an error, never 'no tag')"""
    r = subprocess.run(["git", "-C", root, "rev-parse", "-q", "--verify", f"refs/tags/{tag}^{{commit}}"], capture_output=True)
    if r.returncode not in (0, 1) or (r.returncode == 1 and r.stderr.strip()):
        raise ResearchError(f"git cannot read {root}: {r.stderr.decode('utf-8', 'replace').strip()[:300]}")
    return r.stdout.decode().strip() if r.returncode == 0 else ""


def _index_env(idx: str) -> Dict[str, str]:
    env = dict(os.environ, GIT_INDEX_FILE=idx, GIT_LITERAL_PATHSPECS="1")
    for k in ("GIT_DIR", "GIT_WORK_TREE"):                   # never inherited: they once wrote core.worktree into a shared config
        env.pop(k, None)
    return env


def _tree_with(root: str, paths: Sequence[str], parent: str) -> str:
    """the tree of `parent` with `paths` replaced by their state in the working tree (deletions included), built in a temporary index"""
    fd, idx = tempfile.mkstemp(prefix="research_index_")
    os.close(fd)
    os.remove(idx)
    env = _index_env(idx)
    try:
        git(root, "read-tree", parent, env=env)
        present = [p for p in paths if os.path.exists(os.path.join(root, p))]
        gone = [p for p in paths if p not in present]
        unknown = [p for p in gone if not git(root, "ls-tree", "--name-only", parent, "--", p)]
        if unknown:                                          # neither on disk nor in the parent: a typo or a path relative to elsewhere
            raise ResearchError(f"no such path in {root} (not on disk, not in {parent[:10]}): {unknown}")
        if present:
            git(root, "add", "-A", "--", *present, env=env)
        for p in gone:                                       # a path that is gone from disk: its removal is recorded
            git(root, "rm", "-r", "-q", "--cached", "--ignore-unmatch", "--", p, env=env)
        return git(root, "write-tree", env=env)
    finally:
        if os.path.exists(idx):
            os.remove(idx)


def snapshot(root: str, paths: Sequence[str], message: str, tag: str) -> str:
    """commit the working-tree state of `paths` on top of HEAD (not on any branch) and tag it; -> the commit. An existing tag is an error"""
    if tag_commit(root, tag):
        raise ResearchError(f"the tag {tag} exists already")
    parent = head(root)
    tree = _tree_with(root, paths, parent)
    commit = git(root, "commit-tree", tree, "-p", parent, "-m", message)
    git(root, "tag", "-a", tag, "-m", message.splitlines()[0][:200], commit)
    return commit


def commit_paths(root: str, paths: Sequence[str], message: str) -> str:
    """commit the working-tree state of `paths` onto the current branch; other changes in the working tree and the index stay out.
    -> the new commit, or '' when nothing changed"""
    name = branch(root)
    parent = head(root)
    tree = _tree_with(root, paths, parent)
    if tree == git(root, "rev-parse", f"{parent}^{{tree}}"):
        return ""
    commit = git(root, "commit-tree", tree, "-p", parent, "-m", message)
    git(root, "update-ref", "-m", f"research: {message.splitlines()[0][:80]}", f"refs/heads/{name}", commit, parent)
    for p in paths:                                          # the shared index follows, for these paths only
        if git(root, "ls-tree", "--name-only", commit, "--", p):
            git(root, "reset", "-q", commit, "--", p)
        else:
            git(root, "rm", "-r", "-q", "--cached", "--ignore-unmatch", "--", p)
    return commit


FORBIDDEN = (".pt", ".pth", ".ckpt", ".npy", ".npz", ".pkl", ".h5", ".zip", ".tar", ".gz", ".png", ".jpg", ".jpeg", ".gif", ".mp4", ".webm",
             ".pdf", ".pptx", ".docx", ".html", ".log")
MAX_BYTES = 1_000_000


PRIVATE_FILE = ".private_patterns"                                    # at the top of the checkout, never in git
Patterns = List[Tuple[int, Pattern[str]]]                            # (line of PRIVATE_FILE, the pattern)


def _text(raw: bytes) -> str:
    """bytes as text: UTF-16 with its mark, else UTF-8, else the Windows code page (PowerShell writes all three)"""
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16", "replace")
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode("cp1252", "replace")


def _bytes(root: str, *args: str) -> bytes:
    """what git prints, as it is"""
    r = subprocess.run(["git", "-C", root, *args], capture_output=True)
    if r.returncode != 0:
        raise ResearchError(f"git {' '.join(args[:3])}: {r.stderr.decode('utf-8', 'replace').strip()[:300]}")
    return r.stdout


def private_patterns(root: str) -> Optional[Patterns]:
    """the lines of PRIVATE_FILE as regular expressions (`^` and `$` mean a line's ends; what follows `=>` and lines that start with # are left
    out), from the top of the checkout, else of the main checkout of a worktree; None when there is no such file"""
    common = os.path.join(root, git(root, "rev-parse", "--git-common-dir"))
    places = [os.path.join(d, PRIVATE_FILE) for d in (root, os.path.dirname(os.path.abspath(common)))]
    path = next((p for p in places if os.path.isfile(p)), "")
    if not path:
        return None
    with open(path, "rb") as f:
        lines = _text(f.read()).splitlines()
    out = []
    for n, line in enumerate(lines, 1):
        text = re.split(r"\s+=>(?:\s|$)", line)[0].strip()
        if text and not text.startswith("#"):
            try:
                out.append((n, re.compile(text, re.MULTILINE)))
            except re.error:
                raise ResearchError(f"{PRIVATE_FILE} line {n} is not a regular expression") from None
    return out


def _shown(name: str, patterns: Optional[Patterns]) -> str:
    """a name as it may be printed: withheld when a private pattern matches it"""
    return "(a name that matches a private pattern)" if any(p.search(name) for _, p in patterns or []) else name


def _hits(text: str, where: str, patterns: Patterns) -> List[str]:
    """a line per pattern that matches `text`; the text is never repeated, and `where` must hold no name that was not passed through `_shown`"""
    return [f"{where}: private pattern of line {n}" for n, p in patterns if p.search(text)]


def _private_hits(root: str, commit: str, parents: int, patterns: Patterns) -> Tuple[List[str], List[str]]:
    """(where a commit's author, committer, message, file names or added lines match a private pattern, the files that could not be read as
    text). A merge is read by what it holds that no parent holds"""
    c = commit[:10]
    found = _hits(_text(_bytes(root, "log", "-1", "--format=%an%n%ae%n%cn%n%ce%n%B", commit)), f"{c} author, committer or message", patterns)
    columns = max(parents, 1)
    diff = _bytes(root, "-c", "core.quotepath=false", "diff-tree", "--no-commit-id", "-r", "--root", "-p", "--unified=0", "--no-renames",
                  *(["-c"] if parents > 1 else ["--diff-filter=AM"]), commit)
    unread: List[str] = []
    path, hunk = "", False
    for raw in diff.split(b"\n"):
        if raw.startswith((b"diff --git ", b"diff --combined ", b"diff --cc ")):
            head = raw.decode("utf-8", "replace")
            path, hunk = head.split(" b/", 1)[-1] if head.startswith("diff --git ") else head.split(" ", 2)[-1], False
            found += _hits(path, f"{c} a file name", patterns)
        elif raw.startswith(b"@@"):
            hunk = True
        elif hunk and raw[:columns] == b"+" * columns:
            found += _hits(_text(raw[columns:].rstrip(b"\r")), f"{c} {_shown(path, patterns)}", patterns)
        elif raw.startswith(b"Binary files "):                   # a UTF-16 text is binary to git
            blob = _bytes(root, "cat-file", "blob", f"{commit}:{path}")
            if blob[:2] in (b"\xff\xfe", b"\xfe\xff"):
                found += _hits("\n".join(_text(blob).splitlines()), f"{c} {_shown(path, patterns)}", patterns)
            else:
                unread.append(f"{c} {_shown(path, patterns)}: binary, not scanned for private patterns")
    return found, unread


def unpushed_tags(root: str, remote: str) -> List[str]:
    """the tags (refs/tags/<name>) the remote does not have as they are here; refused when the remote cannot be asked"""
    try:
        r = subprocess.run(["git", "-C", root, "ls-remote", "--tags", remote], capture_output=True, timeout=60,
                           env=dict(os.environ, GIT_TERMINAL_PROMPT="0"))
        why = "" if r.returncode == 0 else r.stderr.decode("utf-8", "replace").strip()[:300]
    except subprocess.TimeoutExpired:
        why = "no answer in 60 s"
    if why:
        raise ResearchError(f"the tags cannot be compared with the remote {remote!r} ({why}), so what they carry is not checked; "
                            "`push-check --no-tags` checks the branch only")
    there = {tuple(line.split("\t", 1)) for line in r.stdout.decode("utf-8", "replace").splitlines() if "\t" in line}
    here = git(root, "for-each-ref", "--format=%(objectname)%09%(refname)", "refs/tags").splitlines()
    return [line.split("\t", 1)[1] for line in here if tuple(line.split("\t", 1)) not in there]


def _remote_of(root: str, upstream: str) -> str:
    """the remote a push goes to: the upstream's, else the branch's, else origin, else the first"""
    remotes = git(root, "remote").split()
    if not remotes:
        raise ResearchError("there is no remote to compare the tags with; `push-check --no-tags` checks the branch only")
    name = git(root, "rev-parse", "--abbrev-ref", upstream, check=False)
    own = git(root, "config", f"branch.{git(root, 'symbolic-ref', '-q', '--short', 'HEAD', check=False)}.remote", check=False)
    for found in [r for r in remotes if name.startswith(r + "/")] + [own, "origin"]:
        if found in remotes:
            return found
    return remotes[0]


def push_problems(root: str, upstream: str = "@{upstream}", tags: bool = True) -> Dict[str, Any]:
    """what a push would publish that never belongs into git: looked at are the commits not yet on the upstream and, for the tags the remote does
    not have, the commits only they carry (a run's snapshot is on no branch). Found: data, weights, pictures, logs, files over 1 MB, a temp/
    folder, the pattern file itself ("forbidden"), and matches of the private patterns, also in the new tags' names and messages and the branch
    name ("private"). No name that matches a pattern is printed"""
    patterns = private_patterns(root)
    commits = [c for c in git(root, "rev-list", f"{upstream}..HEAD").splitlines() if c]
    new_tags: List[str] = []
    tagged: List[str] = []
    if tags:
        remote = _remote_of(root, upstream)
        try:
            new_tags = unpushed_tags(root, remote)
        except ResearchError as e:
            if _shown(str(e), patterns) == str(e):
                raise
            raise ResearchError("the tags cannot be compared with the remote, so what they carry is not checked (the reason is withheld: it matches "
                                "a private pattern); `push-check --no-tags` checks the branch only") from None
        if new_tags:
            known = git(root, "for-each-ref", "--format=%(refname)", f"refs/remotes/{remote}", "refs/tags").splitlines()
            revs = new_tags + [f"^{k}" for k in [git(root, "rev-parse", upstream)] + known if k not in new_tags]
            tagged = [c for c in git(root, "rev-list", "--stdin", data="\n".join(revs).encode()).splitlines() if c and c not in commits]
    bad: List[str] = []
    private: List[str] = []
    unread: List[str] = []
    for c in commits + tagged:
        parents = len(git(root, "rev-list", "--parents", "-n", "1", c).split()) - 1
        which = ["-c"] if parents > 1 else ["--diff-filter=AM"]
        for line in git(root, "diff-tree", "--no-commit-id", "-r", "--root", *which, "-z", "--name-only", c).split("\0"):
            size = git(root, "cat-file", "-s", f"{c}:{line}", check=False) if line else ""
            if not size:                                              # no file there (a merge may remove one)
                continue
            if (line.lower().endswith(FORBIDDEN) or int(size) > MAX_BYTES or "/temp/" in f"/{line}" or line.startswith("temp/")
                    or os.path.basename(line) == PRIVATE_FILE):
                bad.append(f"{c[:10]} {_shown(line, patterns)} ({size} bytes)")
        if patterns is not None:
            found, skipped = _private_hits(root, c, parents, patterns)
            private += found
            unread += skipped
    if patterns is not None:
        private += _hits(git(root, "symbolic-ref", "-q", "--short", "HEAD", check=False), "the branch name", patterns)
        for t in new_tags:
            name = t[len("refs/tags/"):]
            private += _hits(name, "a new tag's name", patterns)
            private += _hits(_text(_bytes(root, "for-each-ref", "--format=%(taggername)%0a%(taggeremail)%0a%(contents)", t)),
                             f"tag {_shown(name, patterns)}", patterns)
    return {"commits": [_shown(git(root, "log", "-1", "--format=%h %s", c), patterns) for c in commits],
            "tags": [_shown(t[len("refs/tags/"):], patterns) for t in new_tags] if tags else "not checked (--no-tags)",
            "commits_only_in_tags": len(tagged), "forbidden": sorted(set(bad)),
            "private": sorted(set(private)) if patterns is not None else f"not scanned: no {PRIVATE_FILE} at the top of the checkout",
            **({"warnings": sorted(set(unread))} if unread else {})}


def show(root: str, commit: str, path: str) -> str:
    """a file as it is in a commit"""
    return git(root, "show", f"{commit}:{path}")


def unpack(root: str, commit: str, dest: str, paths: Sequence[str]) -> None:
    """the files of `paths` at `commit` into `dest` (which must not exist yet)"""
    r = subprocess.run(["git", "-C", root, "archive", "--format=tar", commit, "--", *paths], capture_output=True)
    if r.returncode != 0:
        raise ResearchError(f"git archive {commit[:10]}: {r.stderr.decode('utf-8', 'replace').strip()[:300]}")
    os.makedirs(dest)
    with tarfile.open(fileobj=io.BytesIO(r.stdout)) as tar:
        for member in tar.getmembers():
            if member.name.startswith(("/", "..")) or "/../" in member.name or not (member.isfile() or member.isdir() or member.issym()):
                raise ResearchError(f"git archive {commit[:10]}: refused member {member.name!r}")
        try:
            tar.extractall(dest, filter="data")
        except TypeError:                                    # python before 3.12 has no filter; the members were checked above
            tar.extractall(dest)
