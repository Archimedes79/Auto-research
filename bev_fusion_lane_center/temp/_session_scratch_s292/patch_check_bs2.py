import io
p = "evaluation/jwire_check.py"
s = io.open(p, encoding="utf-8").read()
a = "            assert [c[-1] for c in calls] == [4, 4, 4]"
assert s.count(a) == 1
s = s.replace(a, '''            with mock.patch.object(sys.modules[__name__], "live_check", lambda *x, **k: calls.append(x + (k.get("bs"),)) or (True, 24, {})), contextlib.redirect_stdout(io.StringIO()):
                assert main(["live", "--seed", "1", "--pair-run", "e324_s_s1", "--ends", "att", "--bs", "8"]) == 0
            assert calls[-1][0] == 1 and calls[-1][-1] == 8, calls[-1]                                         # --bs 8 reaches live_check (E358: the dump's batch size)
''' + a.replace("[4, 4, 4]", "[4, 4, 4, 8]"))
io.open(p, "w", encoding="utf-8").write(s)
print("ok")
