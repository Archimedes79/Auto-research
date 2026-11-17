import io
p = "temp/_session_scratch_e358/test_staged.py"
s = io.open(p, encoding="utf-8").read()
a = "def test_cmd_score_refuses_before_anything_long():"
assert s.count(a) == 1
t = '''def test_check_head_derived_run():
    """E363: a run folder whose provenance.json names the head's pair run as `source_pair_run` (the pair run with the node head of B') is a legitimate read-out of the head (a transfer to another graph); an unrelated source, no provenance file, a provenance file without
    the key, the wrong end kind and the wrong seed stay refused"""
    import json
    att = NS(meta=dict(seed=0, pair_run="e324_s_s0", ends="att"))
    with tempfile.TemporaryDirectory() as d, project_root(d):
        for name, prov in (("e363_a_s0", dict(source_pair_run="e324_s_s0")), ("e363_x_s0", dict(source_pair_run="e324_s_s1")), ("e363_nokey_s0", dict(note="x"))):
            os.makedirs(os.path.join(d, "runs", name))
            json.dump(prov, open(os.path.join(d, "runs", name, "provenance.json"), "w"))
        os.makedirs(os.path.join(d, "runs", "e363_noprov_s0"))
        JH.check_head(att, "e363_a_s0", 0, True)                                                          # derived from the head's own pair run: accepted
        JH.check_head(att, "e324_s_s0", 0, True)                                                          # its own pair run: still accepted
        assert "derived from 'e324_s_s1'" in refused(JH.check_head, att, "e363_x_s0", 0, True), "a run derived from ANOTHER pair run"
        assert "pair run" in refused(JH.check_head, att, "e363_noprov_s0", 0, True) and "pair run" in refused(JH.check_head, att, "e363_nokey_s0", 0, True) and "pair run" in refused(JH.check_head, att, "e363_missing_s0", 0, True), "no provenance, no key, no folder"
        assert "trained on 'att' ends" in refused(JH.check_head, att, "e363_a_s0", 0, False), "the end kind is still checked on a derived run"
        assert "seed" in refused(JH.check_head, att, "e363_a_s0", 1, True), "the seed is still checked on a derived run"


'''
s = s.replace(a, t + a)
io.open(p, "w", encoding="utf-8").write(s)
print("ok")
