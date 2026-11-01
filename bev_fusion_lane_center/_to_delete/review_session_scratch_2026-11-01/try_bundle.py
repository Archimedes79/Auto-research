import json, os, sys
sys.path.insert(0, "autoresearch/tools")
import provenance as V
b = V.bundle(sys.argv[1])
for k in ("code_hash", "files", "repo_files", "imports_excluded", "test_imports", "external_local", "unparsed", "dynamic_imports", "missing_roots"):
    print(k, b.get(k))
