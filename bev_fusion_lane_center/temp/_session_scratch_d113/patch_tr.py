import re
import os
p = os.path.join(os.environ["BFLC_ROOT"], "evaluation", "token_readout.py")
s = open(p, encoding="utf-8").read()
old_start = s.index("    rows = [row(0.70")
old_end = s.index("    assert choose_cell([row(0.5")
new = '''    out, a, off, on, low = row(0.70, 0.1, 0.6, False, 0.06), row(0.69, 0.2, 0.6, False, 0.05), row(0.69, 0.3, 0.5, False, 0.04), row(0.69, 0.3, 0.5, True, 0.04), row(0.60, 0.3, 0.5, True, 0.01)
    top, cell, dup = choose_cell([out, a, off, on, low])
    assert top == 0.69 and cell == (0.3, 0.5, True) and dup == 0.04, "the 0.70 cell has dup 0.06 > 0.05 (out); of the three exact ties the one with the record's thresholds AND gate wins, whatever the order of the table"
    assert choose_cell([out, a, off])[1] == (0.3, 0.5, False), "a tie between (0.2, 0.6, off) and (0.3, 0.5, off): the one with the record's thresholds (distance 0) wins"
    assert choose_cell([a, row(0.69, 0.25, 0.5, False, 0.04)])[1] == (0.25, 0.5, False), "the link threshold is compared first (0.05 against 0.1), then the keep threshold"
'''
s = s[:old_start] + new + s[old_end:]
s = s.replace("dict(meta, idx=idx[::2].tolist() + [1] * 0)", "dict(meta, idx=idx[::2].tolist())")
open(p, "w", encoding="utf-8").write(s)
