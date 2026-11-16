import os
R = os.environ["BFLC_ROOT"]


def patch(path, pairs):
    s = open(path, encoding="utf-8").read()
    for old, new in pairs:
        assert s.count(old) == 1, (s.count(old), old[:80])
        s = s.replace(old, new)
    open(path, "w", encoding="utf-8").write(s)


patch(os.path.join(R, "evaluation", "jtest_eval.py"), [
    ('''    """what a dump of `run` made now depends on: the code digest, the device, the checkpoint (a dump is reused only under the same provenance)"""
    return dict(run=run, device=device, code_digest=code_digest(code_sha256()), checkpoint_sha256=sha256_file(os.path.join(ROOT, "runs", run, "model.pt")))''',
     '''    """what a dump of `run` made now depends on: the code digest (and the files behind it, to name what changed), the device, the checkpoint (a dump is reused only under the same provenance)"""
    files = code_sha256()
    return dict(run=run, device=device, code_digest=code_digest(files), checkpoint_sha256=sha256_file(os.path.join(ROOT, "runs", run, "model.pt")), code_sha256=files)'''),
    ('''    bad = [k for k in ("run", "code_digest", "device", "checkpoint_sha256") if got.get(k) != expect.get(k)]
    if bad:
        sys.exit(f"{what}: {path} was made under another {', '.join(bad)} (provenance {hp}): it is not reused; move it away and run again")''',
     '''    bad = [k for k in ("run", "code_digest", "device", "checkpoint_sha256") if got.get(k) != expect.get(k)]
    if bad:
        a, b = got.get("code_sha256") or {}, expect.get("code_sha256") or {}
        files = sorted(f for f in set(a) | set(b) if a.get(f) != b.get(f))
        sys.exit(f"{what}: {path} was made under another {', '.join(bad)} (provenance {hp}" + (f"; changed code files: {', '.join(files[:8])}{' ...' if len(files) > 8 else ''}" if "code_digest" in bad and files else "") + "): it is not reused; move it away and run again")'''),
    ('''    elif expect is not None:
        check_provenance(path, expect, what)
    return load_dump(path, split, what, n)''', '''    elif expect is not None and (make is not None or os.path.exists(head_path(path))):                # a dump of the default path must have its provenance; one named by a flag only when it has a sidecar
        check_provenance(path, expect, what)
    return load_dump(path, split, what, n)'''),
    ('''    (the default paths): an existing one is refused unless its sidecar says the same run, code, device and checkpoint; a dump named by a flag (E321's, the stored VAL dumps: validated, E321's also by hash) has none and is exempt"""''',
     '''    (the default paths): an existing one is refused unless its sidecar says the same run, code, device and checkpoint; a dump named by a flag without a sidecar (E321's, the stored VAL dumps: validated, E321's also by hash) is exempt, one made by
    this module (with a sidecar) is checked like the others"""'''),
    ('''CONTROL_VARIANT = V(''', '''CONTROL_RUN = "e312_p_s{seed}"                                                                       # the control: E312's pair-head run
CONTROL_VARIANT = V('''),
    ('''    dumps = {"b": get_dump(i["b"], (lambda: dump_b(split, run_b, i["b"], a.device, n)) if i["b_made"] else None, split, "B's dump", n, exp(run_b) if i["b_made"] else None)}''',
     '''    dumps = {"b": get_dump(i["b"], (lambda: dump_b(split, run_b, i["b"], a.device, n)) if i["b_made"] else None, split, "B's dump", n, exp(run_b))}'''),
    ('''            dumps["ctl"] = dumps["b"] if i["ctl_b"] == i["b"] else load_dump(i["ctl_b"], split, "the control's B dump", n)
            ends["ctl"] = {f["row"]: f for f in load_dump(i["ctl_ends"], split, "the control's pair dump", n)}''',
     '''            dumps["ctl"] = dumps["b"] if i["ctl_b"] == i["b"] else get_dump(i["ctl_b"], None, split, "the control's B dump", n, exp(run_b))
            ends["ctl"] = {f["row"]: f for f in get_dump(i["ctl_ends"], None, split, "the control's pair dump", n, exp(sub_seed(CONTROL_RUN, seed)))}'''),
])

patch(os.path.join(R, "evaluation", "tests", "test_jtest_eval.py"), [
    ('''            assert p1 == dict(run="r", device="cpu", code_digest=J.code_digest({"a.py": "1"}), checkpoint_sha256=hashlib.sha256(b"weights").hexdigest())''',
     '''            assert p1 == dict(run="r", device="cpu", code_digest=J.code_digest({"a.py": "1"}), checkpoint_sha256=hashlib.sha256(b"weights").hexdigest(), code_sha256={"a.py": "1"}), "the head file names the code files too"'''),
    ('''        assert len(J.get_dump(q, None, "val", "x")) == 873, "a dump named by a flag (E321's, the stored VAL dumps) is exempt"
''', '''        assert len(J.get_dump(q, None, "val", "x", expect=prov)) == 873, "a dump named by a flag without a sidecar (E321's, the stored VAL dumps) is exempt"
        assert len(J.get_dump(p, None, "val", "x", expect=prov)) == 873 and raises(SystemExit, J.get_dump, p, None, "val", "x", expect=dict(prov, device="cuda")), "a dump named by a flag that THIS module made has a sidecar and is checked like the others"
        full_prov = dict(prov, code_sha256={"a.py": "1", "b.py": "1"})
        J.write_head(p, dict(frames=873), full_prov)
        try:
            J.get_dump(p, None, "val", "x", expect=dict(prov, code_digest="c2", code_sha256={"a.py": "2", "b.py": "1"}))
            raise AssertionError("a changed code file must refuse the reuse")
        except SystemExit as e:
            assert "changed code files: a.py" in str(e) and "b.py" not in str(e), str(e)
'''),
])
print("ok")
