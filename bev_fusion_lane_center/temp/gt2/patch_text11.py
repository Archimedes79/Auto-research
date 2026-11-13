p = "temp/jr/report_text.py"
s = open(p, encoding="utf-8").read()


def sub(old, new):
    global s
    assert old in s, old[:70]
    s = s.replace(old, new, 1)


sub('    ["E312", "the pair head (maps + pairs together, handle scales) on the E307 maps, snapped read-out",',
    '    ["E313", "end heat sigma 0.7 m (one step narrower than E307)", "KEEP, both seeds: keypoint F1 0.743 / 0.732 against 0.723 / 0.705, crowded entry recall 0.86 / 0.84 against 0.79; the recall of isolated entries falls by 0.02 / 0.03 and the far range does not move", "<span class=\'pill ok\'>KEEP</span>"],\n'
    '    ["E314", "the E312 pair head on the sigma 0.7 maps", "training (both seeds), result expected ~15:45", "<span class=\'pill info\'>running</span>"],\n'
    '    ["E312", "the pair head (maps + pairs together, handle scales) on the E307 maps, snapped read-out",')
open(p, "w", encoding="utf-8").write(s)
print("ok")
