p = "temp/t21/stream_timing.py"
s = open(p, encoding="utf-8").read()
a = s.index('                    if mode == "light":')
b = s.index('                    else:\n                        r = m(b, dts=dts')
s = s[:a] + '''                    if mode == "light":
                        st["s"] = m.advance(b, st["s"], step if st["s"] else None)
''' + s[b:]
s = s.replace("from precompute import Precompute  # noqa: E402\n", "")
open(p, "w", encoding="utf-8", newline="").write(s)
