"""E379 check (scratch): the topo F1 of A, B and the mean E of seed S on a split with six decimals (the printed diffs of the VAL and the TRAIN run of seed 1 coincide to four decimals: genuine or not?). usage: python ens_digits.py <seed> <split>"""
import sys
sys.argv = [sys.argv[0], sys.argv[1], sys.argv[2]]
src = open(__file__.replace("ens_digits.py", "ens_step0.py"), encoding="utf-8").read().split("print(f\"seed")[0]
exec(compile(src, "ens_step0_head", "exec"))
print(SPLIT, S, "frames", len(frames), "A %.6f B %.6f E %.6f" % (sa["graph"]["topo_F1"], sb["graph"]["topo_F1"], se["graph"]["topo_F1"]), "E-A %.6f E-B %.6f" % (se["graph"]["topo_F1"] - sa["graph"]["topo_F1"], se["graph"]["topo_F1"] - sb["graph"]["topo_F1"]))
