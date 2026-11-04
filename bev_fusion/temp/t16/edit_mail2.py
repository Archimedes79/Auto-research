import re
p = "bev_fusion/temp/t16/build_mail2.py"
s = open(p, encoding="utf-8").read()
a = s.index('    out.append("=== WHAT WAS TRIED')
b = s.index('    out.append("=== AGAINST THE STATE OF THE ART')
s = s[:a] + s[b:]
s = s.replace('=== AGAINST THE STATE OF THE ART (same frames', '4. AGAINST THE STATE OF THE ART (same frames')
s = s.replace('=== RESULTS (validation set', '=== TABLE (validation set')
s = s.replace('''    out.append(f"Repository: github.com''', '''    end = "bev_fusion/temp/t16/mail_narrative2_end.txt"
    out.append(open(end, encoding="utf-8").read().rstrip("\n") if os.path.exists(end) else "(closing narrative missing)")
    out.append("")
    out.append(f"Repository: github.com''')
open(p, "w", encoding="utf-8").write(s)
