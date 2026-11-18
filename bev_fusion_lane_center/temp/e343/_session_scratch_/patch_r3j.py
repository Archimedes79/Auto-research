import os
R = os.environ["BFLC_ROOT"]


def patch(path, pairs):
    s = open(path, encoding="utf-8").read()
    for old, new in pairs:
        assert s.count(old) == 1, (s.count(old), old[:80])
        s = s.replace(old, new)
    open(path, "w", encoding="utf-8").write(s)


patch(os.path.join(R, "evaluation", "jtest_eval.py"), [
    ('''@contextlib.contextmanager
def atomic(path, overwrite=True):''', '''def check_writable(path):
    """refuses a path under the deny-list (E321's folders) and a path outside the allow-list (the WRITABLE folders): before anything is created, and at the start of a dump, not after its 10 minutes"""
    if is_protected(path):
        raise PermissionError(f"refused: {path} lies under {' / '.join(PROTECTED)} (the artefacts of E321 are read-only inputs)")
    if not is_writable(path):
        raise PermissionError(f"refused: {path} is not under {' / '.join(WRITABLE)} (the only folders this module writes)")


@contextlib.contextmanager
def atomic(path, overwrite=True):'''),
    ('''    if is_protected(path):
        raise PermissionError(f"refused: {path} lies under {' / '.join(PROTECTED)} (the artefacts of E321 are read-only inputs)")
    if not is_writable(path):
        raise PermissionError(f"refused: {path} is not under {' / '.join(WRITABLE)} (the only folders this module writes)")
    if not overwrite and os.path.exists(path):''', '''    check_writable(path)
    if not overwrite and os.path.exists(path):'''),
    ('''    head_path(out)                                                                                     # a dump name ends with .pkl; an existing dump is never overwritten
    refuse_existing(out)''', '''    head_path(out)                                                                                     # a dump name ends with .pkl, lies where this module writes, and an existing dump is never overwritten
    check_writable(out)
    refuse_existing(out)'''),
    ('''    head_path(out)
    refuse_existing(out)
    prov = provenance(run, device)                                                                     # at the start, as in dump_b''', '''    head_path(out)
    check_writable(out)
    refuse_existing(out)
    prov = provenance(run, device)                                                                     # at the start, as in dump_b'''),
    ('''        sys.exit(f"STOP: the imported lane code (digest {digest}) is not the code of the rehearsal (--expect-code-digest {a.expect_code_digest}){named or '; give --expect-code-card <the rehearsal card> to name the files'}")''',
     '''        sys.exit(f"STOP: the imported lane code (digest {digest}) is not the code of the rehearsal (--expect-code-digest {a.expect_code_digest}){named or '; give --expect-code-card <the rehearsal card> to name the files'}"
                 + ("" if not a.expect_code_card or changed else " (the card's file hashes equal the current ones: that digest is not this card's)"))'''),
    ('''        named = ""
        if a.expect_code_card:''', '''        named, changed = "", []
        if a.expect_code_card:'''),
])
patch(os.path.join(R, "evaluation", "tests", "test_jtest_eval.py"), [
    ('''        assert raises(RuntimeError, J.dump_b, "val", "e273_p_s0", os.path.join(d, "x.out")) and raises(RuntimeError, J.dump_ends, "val", "e324_s_s0", os.path.join(d, "y")), "no .pkl: the head file would take the name"
''', '''        assert raises(RuntimeError, J.dump_b, "val", "e273_p_s0", os.path.join(d, "x.out")) and raises(RuntimeError, J.dump_ends, "val", "e324_s_s0", os.path.join(d, "y")), "no .pkl: the head file would take the name"
        assert raises(PermissionError, J.dump_b, "val", "e273_p_s0", os.path.join(d, "z.pkl")) and raises(PermissionError, J.dump_ends, "val", "e324_s_s0", os.path.join(d, "z.pkl")), "a place outside the allow-list: refused at the START of the dump, not after it"
'''),
    ('''            assert (f"changed files against {kw['expect_code_card']}: " + ", ".join(names) in msg) if names else ("--expect-code-card" in msg), msg
''', '''            assert (f"changed files against {kw['expect_code_card']}: " + ", ".join(names) in msg) if names else ("--expect-code-card" in msg), msg
    with tempfile.TemporaryDirectory() as d:                                                           # a card whose file hashes are the current ones: the digest given is not that card's
        json.dump(dict(code_sha256=files), open(os.path.join(d, "same.json"), "w"))
        rc, *_ = toy_cmd_all(d, with_(expect_code_digest="0" * 64, expect_code_card=os.path.join(d, "same.json")), codes=[files, files])
        assert isinstance(rc, SystemExit) and "equal the current ones" in str(rc), str(rc)
'''),
])
print("ok")
