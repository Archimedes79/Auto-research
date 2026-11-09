p = "evaluation/bev_gtcheck.py"
t = open(p, encoding="utf-8").read()
a = '''    print(f"bev_gtcheck selftest ok ({len(samples)} real frames, 3 dt each)")
    print_table(rows)
'''
b = '''    iou, dev = alignment(samples)
    for layer, floor in (("drivable_area", 0.95), ("walkway", 0.9), ("road_divider", 0.5)):
        for dt in (-0.5, 0.5):
            assert iou[(layer, dt, "warped")] > max(floor, iou[(layer, dt, "unwarped")] + 0.1) - 1e-9 or layer == "road_divider" and iou[(layer, dt, "warped")] > 2 * iou[(layer, dt, "unwarped")], (layer, dt, iou)
    assert dev < 0.2, f"the displacement of the poses and of meta['steps'] differ by {dev:.3f} m"
    print(f"bev_gtcheck selftest ok ({len(samples)} real frames, 3 dt each)")
    print_table(rows)
    print_alignment(iou, dev)
'''
assert a in t
t = t.replace(a, b)
a2 = "def selftest(n=24):"
b2 = '''def print_alignment(iou, dev):
    print("\nGT frames against their poses (static layers of cur warped into the ego frame of prev / next, IoU of the binary maps where both exist):")
    for layer in STATIC:
        print(f"  {layer:14s}" + "".join(f"  dt{dt:+.1f}: unwarped {iou[(layer, dt, 'unwarped')]:.3f} -> warped {iou[(layer, dt, 'warped')]:.3f}" for dt in (-0.5, 0.5)))
    print(f"  displacement of the recorded poses against the one integrated from meta['steps']: mean abs difference {dev:.3f} m")


def selftest(n=24):'''
t = t.replace(a2, b2, 1)
a3 = '''        print_table(rows)

'''
t = t.replace('''        print(f"{len(samples)} real GT frames (cur / prev / next), signature {a.sig}")
        print_table(rows)''', '''        print(f"{len(samples)} real GT frames (cur / prev / next), signature {a.sig}")
        print_table(rows)
        print_alignment(*alignment(samples))''')
open(p, "w", encoding="utf-8", newline="\n").write(t)
print("patched")
