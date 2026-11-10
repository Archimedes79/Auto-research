s = open("temp/pitch/patch_sets2.py", encoding="utf-8").read()
head = s[:s.index('patch("evaluation/bev_sets.py"')]
tail = s[s.index('patch("evaluation/bev_suite.py"'):]
tail = tail.replace("""('''Global options: --device cuda|cpu  --workers N  --split val|train  --ckpt-dir DIR''', '''Global options: --device cuda|cpu  --workers N  --split val|train|<split>.<part>  --ckpt-dir DIR'''),""",
                    """('''--split val|train  --ckpt-dir DIR (moved checkpoints)''', '''--split val|train|<split>.<part>  --ckpt-dir DIR (moved checkpoints)'''),""")
open("temp/pitch/patch_sets3.py", "w", encoding="utf-8").write(head + tail)
