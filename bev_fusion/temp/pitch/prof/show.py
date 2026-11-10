import io
import pstats
s = io.StringIO()
p = pstats.Stats("temp/pitch/prof/eval.prof", stream=s)
p.sort_stats("cumulative").print_stats(70)
print(s.getvalue()[:12000])
