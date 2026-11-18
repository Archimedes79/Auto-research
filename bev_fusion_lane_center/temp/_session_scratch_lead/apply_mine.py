import sys
p = "check_git_head.py"
s = open(p, encoding="utf-8").read()
def rep(old, new, count=1):
    global s
    assert s.count(old) == count, (s.count(old), old[:60])
    s = s.replace(old, new)
rep('MAX_BYTES = 1_000_000\nSCRATCH_DIRS = ("temp", "_to_delete")',
    'MAX_BYTES = 1_000_000\nRECORD_NAMES = {"LEDGER.csv", "LEDGER.md"}                                              # the run ledgers are one complete text record each (CLAUDE.md: do not split or roll them)\nRECORD_MAX_BYTES = 3_000_000                                                            # ... and grow with every run: their limit is 3 MB (D120 follow-up, 2026-11-18: LEDGER.csv passed 1 MB)\nSCRATCH_DIRS = ("temp", "_to_delete")')
rep('def classify(path, size=None):\n    """\'scratch\' | \'forbidden\' | \'ok\' for a repository-relative path (size in bytes when known)"""',
    'def size_limit(path):\n    """the size limit of a repository-relative path: 1 MB, 3 MB for the run ledgers (RECORD_NAMES)"""\n    return RECORD_MAX_BYTES if os.path.basename(path.replace("\\\\", "/")) in RECORD_NAMES else MAX_BYTES\n\n\ndef classify(path, size=None):\n    """\'scratch\' | \'forbidden\' | \'ok\' for a repository-relative path (size in bytes when known)"""')
rep('    return "forbidden" if size is not None and size > MAX_BYTES else "ok"', '    return "forbidden" if size is not None and size > size_limit(path) else "ok"')
rep('if kind == "forbidden" and not (size or 0) > MAX_BYTES and "__pycache__" not in path', 'if kind == "forbidden" and not (size or 0) > size_limit(path) and "__pycache__" not in path')
rep('("forbidden" if sizes.get(blob, 0) > MAX_BYTES else "ok")      # a change of an old scratch file is legacy', '("forbidden" if sizes.get(blob, 0) > size_limit(path) else "ok")      # a change of an old scratch file is legacy')
rep('f" ({sizes[blob] // 1000} kB)" if sizes.get(blob, 0) > MAX_BYTES else ""', 'f" ({sizes[blob] // 1000} kB)" if sizes.get(blob, 0) > size_limit(path) else ""')
rep('    assert classify("a/big.json", 2_000_000) == "forbidden" and classify("a/small.json", 10) == "ok" and classify("a/__pycache__/x.py") == "forbidden"',
    '    assert classify("a/big.json", 2_000_000) == "forbidden" and classify("a/small.json", 10) == "ok" and classify("a/__pycache__/x.py") == "forbidden"\n    assert classify("p/experiment_execution/LEDGER.csv", 1_052_009) == "ok" and classify("p/LEDGER.md", 2_900_000) == "ok"                  # the run ledgers: 3 MB, not 1 MB\n    assert classify("p/LEDGER.csv", 3_000_001) == "forbidden" and classify("p/LEDGER.csv.bak", 1_100_000) == "forbidden" and classify("p/other.csv", 1_052_009) == "forbidden"')
rep('anything over 1 MB; a', 'anything over 1 MB (3 MB for the run ledgers LEDGER.csv / LEDGER.md); a')
open(p, "w", encoding="utf-8", newline="").write(s)
print("ok")
