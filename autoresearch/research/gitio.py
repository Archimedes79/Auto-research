"""git, the few ways the research system uses it: a snapshot commit of the working tree through a temporary index (the shared index and the
branch stay as they are) with an annotated tag; a commit of named paths onto the branch (what other chats edit stays out); reading a file
of a commit; unpacking a commit into a folder."""
from __future__ import annotations

import io
import os
import subprocess
import tarfile
import tempfile
from typing import Dict, List, Optional, Sequence

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
    """the commit a tag points at, '' when there is no such tag"""
    return git(root, "rev-parse", "-q", "--verify", f"refs/tags/{tag}^{{commit}}", check=False)


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


def push_problems(root: str, upstream: str = "@{upstream}") -> Dict[str, List[str]]:
    """what the commits not yet on the upstream add that never belongs into git: data, weights, pictures, logs, files over 1 MB, a temp/ folder"""
    commits = [c for c in git(root, "rev-list", f"{upstream}..HEAD").splitlines() if c]
    bad: List[str] = []
    for c in commits:
        for line in git(root, "diff-tree", "--no-commit-id", "-r", "--diff-filter=AM", "-z", "--name-only", c).split("\0"):
            if not line:
                continue
            size = int(git(root, "cat-file", "-s", f"{c}:{line}") or 0)
            if line.lower().endswith(FORBIDDEN) or size > MAX_BYTES or "/temp/" in f"/{line}" or line.startswith("temp/"):
                bad.append(f"{c[:10]} {line} ({size} bytes)")
    return {"commits": [git(root, "log", "-1", "--format=%h %s", c) for c in commits], "forbidden": sorted(set(bad))}


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


def changed_files(root: str, a: str, b: str, paths: Sequence[str]) -> List[str]:
    out = git(root, "diff", "--name-only", a, b, "--", *paths)
    return [x for x in out.splitlines() if x]


def diff(root: str, a: str, b: str, paths: Sequence[str]) -> bytes:
    r = subprocess.run(["git", "-C", root, "diff", "--binary", a, b, "--", *paths], capture_output=True)
    if r.returncode != 0:
        raise ResearchError(f"git diff: {r.stderr.decode('utf-8', 'replace').strip()[:300]}")
    return r.stdout


def apply(root: str, patch: bytes, check_only: bool = False) -> None:
    """apply a patch to the working tree (three-way where the base is known); a conflict is an error and leaves the tree as it was"""
    if not patch.strip():
        return
    git(root, "apply", "--3way", "--check", data=patch)
    if not check_only:
        git(root, "apply", "--3way", data=patch)
