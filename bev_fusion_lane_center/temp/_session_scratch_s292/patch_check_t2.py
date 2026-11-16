import io
p = "evaluation/jwire_check.py"
s = io.open(p, encoding="utf-8").read()
a = '''                assert main(["live", "--seed", "1", "--pair-run", "e324_s_s1", "--ends", "att", "--faithful"]) == 0
'''
assert s.count(a) == 1
s = s.replace(a, a + '''                assert main(["live", "--seed", "1", "--pair-run", "e324_s_s1", "--ends", "att", "--faithful", "--bs", "8"]) == 2          # the faithful pass fixes the batch sizes: --bs with it is refused before anything runs
''')
b = '''            assert [c[-2] for c in calls] == [4, 4, 4, 8, 4]'''
assert s.count(b) == 1
io.open(p, "w", encoding="utf-8").write(s)
print("ok")
