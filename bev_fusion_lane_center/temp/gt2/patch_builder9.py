p = "temp/jr/build_junction_report.py"
s = open(p, encoding="utf-8").read()


def sub(old, new):
    global s
    assert old in s, old[:70]
    s = s.replace(old, new, 1)


sub("{FAIL}{maneuver_block()}<h3>After: E305 pairs, connector ends snapped, p >= 0.1</h3>{errors_block('errors_snap', 'aft_', 'After (E305, snapped, p >= 0.1). ')}", "{FAIL}{FAIL2}{maneuver_block()}<h3>After: E303 pairs with learned handle scales, connector ends snapped, p >= 0.1</h3>{errors_block('errors_snap_e303', 'aft_', 'After (E303, snapped, p >= 0.1). ')}")
open(p, "w", encoding="utf-8").write(s)
