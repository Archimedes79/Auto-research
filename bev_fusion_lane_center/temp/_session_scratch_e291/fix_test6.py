p = "temp/e291/test_e291_staged.py"
s = open(p, encoding="utf-8").read()
a = s.index("    for env, ok in ((\"1.0\", True), (None, False), (\"1.5\", False)):")
b = s.index("    print(\"bfsig ok\")")
new = '''    for env, ok in (("1.0", True), (None, False), ("1.5", False)):
        if env is None:
            os.environ.pop("BFLC_SIG", None)
        else:
            os.environ["BFLC_SIG"] = env
        stopped = None
        try:
            fake_main(JT, ["--name", "smoke_e291", "--maps", ck, "--e2e", "--scales", "--bs", "2", "--workers", "0"], steps=1, record={})
        except AssertionError as e:
            stopped = str(e)
        assert (stopped is None) == ok and (ok or ("BFLC_SIG" in stopped and "maps checkpoint" in stopped)), f"BFLC_SIG={env} with sigma-1.0 maps: ok={ok}, stopped with {stopped!r}"
'''
s = s[:a] + new + s[b:]
open(p, "w", encoding="utf-8", newline="\n").write(s)
print("ok")
