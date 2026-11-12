p = "temp/jr/report_text.py"
s = open(p, encoding="utf-8").read()
old = "\nGT = (\"<p>The layer is made on the lane side"
assert old in s
add = '''
ARCH2 = ("<div class='note'><b>The delivered system after the day (E312).</b> Frozen GaussianCaR features and B's pyramid -> junction maps head with end heat targets of sigma 1.0 m (E307) -> keypoints (3 x 3 peaks, soft-argmax) -> pair head with order features, evidence along the curve and two learned handle scales per pair (E303), trained TOGETHER with the maps head (E305: the pair loss reaches the heat through the soft-argmax positions, the pyramid stays frozen) -> connectors at p >= 0.1 whose ends are snapped to the delivered lane pieces: the keypoints decide which entry joins which exit, the curve itself starts at the end of the nearest delivered piece wired to the entry and ends at the start of the nearest piece wired to the exit (E311, one piece per end). "
         "The only hard steps are the peak choice and the read-out threshold; everything the loss sees is soft.</div>")
'''
s = s.replace(old, add + old, 1)
open(p, "w", encoding="utf-8").write(s)
p = "temp/jr/build_junction_report.py"
s = open(p, encoding="utf-8").read()
old = "{architecture()}{ARCH}"
assert old in s
s = s.replace(old, "{architecture()}{ARCH}{ARCH2}", 1)
open(p, "w", encoding="utf-8").write(s)
print("ok")
