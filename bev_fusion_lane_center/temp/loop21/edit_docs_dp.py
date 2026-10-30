def sub(p, a, b):
    t = open(p, encoding="utf-8").read()
    assert t.count(a) == 1, (p, a[:50])
    open(p, "w", encoding="utf-8").write(t.replace(a, b))
sub("evaluation/README.md", "`test_p16_spec.py` (18 spec tests of the builder on the numpy reference) |", "`test_p16_spec.py` (18 spec tests of the builder on the numpy reference); `test_dp_simplify.py` (6 spec tests of the output stage `simplify_dp`, toy graphs) |")
sub("tools/maintenance/run_all_tests.sh", 'run "builder spec tests (18)"               test_p16_spec.py\n', 'run "builder spec tests (18)"               test_p16_spec.py\nrun "output stage simplify_dp (6)"          test_dp_simplify.py\n')
