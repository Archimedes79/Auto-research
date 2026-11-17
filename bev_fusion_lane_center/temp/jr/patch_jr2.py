def sub(path, old, new, count=1):
    s = open(path, encoding="utf-8").read()
    assert old in s, (path, old[:90])
    open(path, "w", encoding="utf-8").write(s.replace(old, new, count))
t = "report_text.py"
sub(t, '"far range (> 35 m) and Singapore; paint and visibility matter little"', '"far range (> 35 m: recall 0.23 against 0.38 near) and Singapore (0.24 against 0.35); unpainted ends are found 8 - 12 points less often, visibility does not matter (strict matching, two seeds)"')
sub(t, '''    ["E291 / E292 / E294 / E295 / E297 / E298 / E299 / E302 / E303", "road-level lane counts, Matrix-NMS layer, nodes from 1 m maps, sigma head, chord keep-gate, driven weight, time prior, capacitated assignment, handle scales", "cards written, behind the night runs", "<span class='pill info'>planned</span>"],''',
    '''    ["E298 / E303 / E304", "driven pairs weighted higher, two learned handle scales per pair (the spline), far-range weighted maps", "code written and tested, queued behind the night chain (day chain 1)", "<span class='pill info'>queued</span>"],
    ["E302 / E305 / E306", "capacitated assignment with degree heads (entry may feed several exits), maps + pair head trained together (the pair loss reaches the heat), exact polygon targets (the legacy fill is half a cell off)", "code written and tested on real frames (CPU), queued (day chains 2 and 3)", "<span class='pill info'>queued</span>"],
    ["E291 / E292 / E294 / E295 / E297 / E299", "road-level lane counts, Matrix-NMS layer, nodes from 1 m maps, sigma head, chord keep-gate, time prior", "cards written", "<span class='pill info'>planned</span>"],''')
print("ok")
