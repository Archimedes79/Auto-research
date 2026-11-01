"""one-off: the result package of a run lives in <problem>/results/<run id>/ (provenance.json, bundle/, package.json)"""
import sys


def patch(path, pairs):
    s = open(path, encoding="utf-8").read()
    for old, new, n in pairs:
        assert s.count(old) == n, (path, old[:70], s.count(old))
        s = s.replace(old, new)
    open(path, "w", encoding="utf-8", newline="\n").write(s)
    print("ok", path)


patch("autoresearch/tools/provenance.py", [
    ('    pj = os.path.join(os.path.dirname(rj), "provenance.json")', '    pj = os.path.join(results_dir(run_id), "provenance.json")', 1),
    ('    out = os.path.join(os.path.dirname(rj), "provenance.json")', '    out = os.path.join(results_dir(run_id), "provenance.json")', 1),
    ('into=os.path.join(run_dir, "bundle") if write else None', 'into=os.path.join(results_dir(run_id), "bundle") if write else None', 1),
    ('        P.atomic_write(out, json.dumps(block, indent=1) + "\\n")', '        os.makedirs(os.path.dirname(out), exist_ok=True)\n        P.atomic_write(out, json.dumps(block, indent=1) + "\\n")', 1),
    ('-> <run_records>/<id>/provenance.json', '-> <problem>/results/<id>/provenance.json', 1),
    ('(runs/<id>/bundle/)', '(results/<id>/bundle/)', 1),
    ('Written to <run_records>/<id>/provenance.json (`repro` reads it)', 'Written to the result folder <problem>/results/<id>/provenance.json (`repro` reads it)', 1),
    ("os.path.join(P.path('run_records'), a.args[0], 'provenance.json')", "os.path.join(results_dir(a.args[0]), 'provenance.json')", 1),
    ('def find_run_dir(run_id):',
     'def results_dir(run_id):\n    """the result folder of a run: <problem>/<project.json "results", default results>/<run id>. Text records go to git (package.json, provenance.json);\n    the heavy files stay in the run folder and are only listed in the package"""\n    return os.path.join(P.root(), P.cfg_of(P.root()).get("results", "results"), run_id)\n\n\ndef find_run_dir(run_id):', 1),
    ('put(tr, "bundle/stale.py", "old\\n")', 'put(os.path.join(proj, "results", "r2"), "bundle/stale.py", "old\\n")', 1),
    ('os.path.join(tr, "bundle", "pkg", "head.py")', 'os.path.join(proj, "results", "r2", "bundle", "pkg", "head.py")', 1),
    ('os.path.join(tr, "bundle", "config.json")', 'os.path.join(proj, "results", "r2", "bundle", "config.json")', 1),
    ('os.path.join(tr, "bundle", "data.py")', 'os.path.join(proj, "results", "r2", "bundle", "data.py")', 1),
    ('os.path.join(proj, "rr", "r2", "provenance.json")', 'os.path.join(proj, "results", "r2", "provenance.json")', 1),
    ('os.path.join(tr, "bundle", "stale.py")', 'os.path.join(proj, "results", "r2", "bundle", "stale.py")', 1),
])
patch("autoresearch/tools/package.py", [
    ('def package_path(run_id):\n    return os.path.join(P.path("run_records"), run_id, "package.json")', 'def package_path(run_id):\n    return os.path.join(V.results_dir(run_id), "package.json")', 1),
    ('    prov_file = os.path.join(P.path("run_records"), run_id, "provenance.json")', '    prov_file = os.path.join(V.results_dir(run_id), "provenance.json")', 1),
    ('    P.atomic_write(package_path(run_id), json.dumps(pk, indent=1) + "\\n")\n    return pk', '    os.makedirs(os.path.dirname(package_path(run_id)), exist_ok=True)\n    P.atomic_write(package_path(run_id), json.dumps(pk, indent=1) + "\\n")\n    return pk', 1),
    ('    python autoresearch/tools/package.py package <run id> [--run-dir d]   write <run records>/<id>/package.json (a text record, in git)', '    python autoresearch/tools/package.py package <run id> [--run-dir d]   write <problem>/results/<id>/package.json (a text record, in git)', 1),
    ('The package is the one place that says', 'The result folder <problem>/results/<run id>/ holds the package: package.json, provenance.json and the copy of the model bundle (bundle/, not in git:\nthe code is in the tag). The package is the one place that says', 1),
    ('(list in records/runs/{rid}/package.json)', '(list in results/{rid}/package.json)', 1),
])
