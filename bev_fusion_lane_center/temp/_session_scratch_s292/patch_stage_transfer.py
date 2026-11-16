import io
p = "temp/_session_scratch_e358/stage/jwire_head.py"
s = io.open(p, encoding="utf-8").read()
a = '''    hp = m.get("pair_run") or f"e312_p_s{hs}"
    assert hp == run, f"the head belongs to the pair run {hp!r}, the read-out is on {run!r}: refused"
'''
assert s.count(a) == 1
b = '''    hp = m.get("pair_run") or f"e312_p_s{hs}"
    if hp != run:                                                                                          # E363: a run folder DERIVED from the head's pair run (its provenance.json names it as `source_pair_run`: the pair run with the node head of B') is a legitimate read-out of the head (transfer to another graph)
        import json
        root = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        pv = os.path.join(root, "runs", run, "provenance.json")
        src = json.load(open(pv)).get("source_pair_run") if os.path.exists(pv) else None
        assert src == hp, f"the head belongs to the pair run {hp!r}, the read-out is on {run!r}" + ("" if src is None else f" (derived from {src!r})") + ": refused"
'''
s = s.replace(a, b)
if "\nimport os" not in s:
    s = s.replace("import numpy as np\n", "import os\nimport numpy as np\n", 1)
io.open(p, "w", encoding="utf-8").write(s)
print("ok")
