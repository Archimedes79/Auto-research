import time, numpy as np
t0 = time.time()
import p16_data as PD
bd = PD.BuilderData("val_clean")
print("BuilderData", time.time() - t0, len(bd.rows))
import qgraph_eval as QE
fr = QE.val_frames()
print("rows equal:", [f.row for f in fr] == [int(r) for r in bd.rows], bd.rows[:5])
