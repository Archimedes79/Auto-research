import os, sys
ROOT = os.getcwd()
for sub in ("evaluation", "evaluation/tests"):
    sys.path.insert(0, os.path.join(ROOT, sub))
import test_jtest_eval as T
for name in ("test_the_code_pin_cannot_be_bypassed", "test_standalone_dumps_are_refused_on_test_even_with_every_gate_open", "test_the_test_line_is_fixed_by_the_card"):
    getattr(T, name)()
    print("pass", name)
