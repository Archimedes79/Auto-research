import sys
p = sys.argv[1]
s = open(p, encoding="utf-8").read()
a = 'print("numbers: layer-1 comparison at 1e-6 (9e-7 passes, 2e-6 fails, a missing key fails), g1, the samples, the parity comparison inside / outside its tolerance: ok")'
assert s.count(a) == 1
s = s.replace(a, '''print("numbers: layer-1 comparison at 1e-6, g1, the samples, parity inside / outside its tolerance, the matching of two token sets by cell, the unmatched-token limit (a perturbed live set FAILs), gpu_free without this process and its parents, "
          "stamped result files, the dump identity, control card names with K, layer 1 stores plain B only on a PASS: ok")''')
open(p, "w", encoding="utf-8").write(s)
