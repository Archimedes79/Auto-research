"""one-off: the repository files of a bundle are read from the run's code tag (git show tag:path), not from today's working tree"""
p = "autoresearch/tools/provenance.py"
s = open(p, encoding="utf-8").read()


def sub(old, new):
    global s
    assert s.count(old) == 1, (old[:70], s.count(old))
    s = s.replace(old, new)


# 1. repo_modules: an existence test that can ask a tag
sub('''def repo_modules(dotted_names, repo):''', '''def git_bytes(repo, ref, rel):
    """the content of <ref>:<rel> as bytes, None when the tag has no such file"""
    try:
        r = subprocess.run(["git", "--no-optional-locks", "-C", repo, "show", f"{ref}:{rel}"], capture_output=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout if r.returncode == 0 else None


def repo_modules(dotted_names, repo, ref=None):''')
sub('''    (torch.nn, numpy) find nothing"""
    out = set()''', '''    (torch.nn, numpy) find nothing. With `ref` (a tag) the files are looked for in that tag, not in the working tree"""
    out = set()

    def has(rel):
        return git_bytes(repo, ref, rel.replace("\\\\", "/")) is not None if ref else os.path.isfile(os.path.join(repo, rel))''')
sub('''            f = base + ".py" if os.path.isfile(base + ".py") else (os.path.join(base, "__init__.py") if os.path.isfile(os.path.join(base, "__init__.py")) else None)
            if f:
                out.add(os.path.relpath(f, repo).replace("\\\\", "/"))
                for j in range(1, i):
                    init = os.path.join(repo, *parts[:j], "__init__.py")
                    if os.path.isfile(init):
                        out.add(os.path.relpath(init, repo).replace("\\\\", "/"))
                break''', '''            rels = ["/".join(parts[:i]) + ".py", "/".join(parts[:i]) + "/__init__.py"]
            f = next((r for r in rels if has(r)), None)
            if f:
                out.add(f)
                for j in range(1, i):
                    init = "/".join(parts[:j]) + "/__init__.py"
                    if has(init):
                        out.add(init)
                break''')
# 2. bundle: repo_ref
sub('''def bundle(code_dir, roots=BUNDLE_ROOTS, into=None, config=None, exclude=BUNDLE_EXCLUDE):''', '''def bundle(code_dir, roots=BUNDLE_ROOTS, into=None, config=None, exclude=BUNDLE_EXCLUDE, repo_ref=None):''')
sub('''    named by an import that no file was found for), dynamic_imports; the imports of the repository files themselves are not followed.''', '''    named by an import that no file was found for), dynamic_imports; the imports of the repository files themselves are not followed. repo_ref = the
    code tag of the run: the repository files are taken from that tag (git show), so they are the ones the run saw, not today's.''')
sub('''    repo_files = repo_modules(info["external_mods"], repo)''', '''    repo_files = repo_modules(info["external_mods"], repo, repo_ref)''')
sub('''    items = [(rel, os.path.join(code_dir, rel)) for rel in files] + [("_repo/" + rel, os.path.join(repo, rel)) for rel in repo_files]''', '''    keep_tmp = tempfile.mkdtemp() if repo_ref else None      # the files of the tag, written out so that they can be hashed and copied like the others
    repo_items = []
    for rel in repo_files:
        if repo_ref:
            full = os.path.join(keep_tmp, rel)
            os.makedirs(os.path.dirname(full), exist_ok=True)
            open(full, "wb").write(git_bytes(repo, repo_ref, rel) or b"")
        else:
            full = os.path.join(repo, rel)
        repo_items.append(("_repo/" + rel, full))
    items = [(rel, os.path.join(code_dir, rel)) for rel in files] + repo_items''')
sub('''           "source": os.path.basename(os.path.dirname(code_dir.rstrip("/\\\\"))) + "/" + os.path.basename(code_dir.rstrip("/\\\\"))}
    if config and os.path.isfile(config):
        out["config_hash"] = file_hash(config)
    if into:''', '''           "source": os.path.basename(os.path.dirname(code_dir.rstrip("/\\\\"))) + "/" + os.path.basename(code_dir.rstrip("/\\\\")), "repo_files_from": repo_ref or "the working tree"}
    if config and os.path.isfile(config):
        out["config_hash"] = file_hash(config)
    if into:''')
sub('''                out["copied_to"] = os.path.basename(into)
    return out''', '''                out["copied_to"] = os.path.basename(into)
    if keep_tmp:
        shutil.rmtree(keep_tmp, ignore_errors=True)
    return out''')
# 3. manifest passes the tag when it exists here
sub('''        block["bundle"] = bundle(snap, into=os.path.join(results_dir(run_id), "bundle") if write else None, config=cfg)''', '''        tag = r.get("tag")
        ref = tag if tag and git(repo_root(), "rev-parse", "--verify", "--quiet", f"refs/tags/{tag}") is not None else None
        block["bundle"] = bundle(snap, into=os.path.join(results_dir(run_id), "bundle") if write else None, config=cfg, repo_ref=ref)''')
# 4. selftest: the file is read from a tag, not from the working tree
sub('''        finally:
            globals()["repo_root"] = real_root3''', '''            run("add", "autogt")
            run("commit", "-q", "-m", "autogt")
            run("tag", "tg/layers")
            put(tmp, "autogt/layers.py", "AREAS = 99  # edited after the run\\n")
            put(s3, "model.py", "import autogt.layers\\n")
            put(s3, "losses.py", "Y = 1\\n")
            bt = bundle(s3, into=os.path.join(tmp, "out4"), repo_ref="tg/layers")
            assert bt["repo_files"] == ["autogt/__init__.py", "autogt/layers.py"] and bt["repo_files_from"] == "tg/layers", bt
            assert open(os.path.join(tmp, "out4", "_repo", "autogt", "layers.py")).read() == "AREAS = 1\\n", "the copy is the file of the tag, not today's"
            assert bundle(s3, repo_ref="tg/layers")["code_hash"] != bundle(s3)["code_hash"], "the tag's file and the edited working file differ in the hash"
            assert bundle(s3, repo_ref="tg/layers")["code_hash"] == bt["code_hash"] and bundle(s3, repo_ref="tg/nonexistent")["repo_files"] == [], "a ref without the files finds none"
        finally:
            globals()["repo_root"] = real_root3''')
open(p, "w", encoding="utf-8", newline="\n").write(s)
print("ok")
