import os, sys, pickle
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "loop20"))
import evaluate as EV
import p16_data as PD
print(EV.SPLITS)
ds = PD.BuilderData(EV.SPLITS["val"][0], ks=None, train=False)
print(len(ds), ds.part, ds.rows[:5])
ds._open()
print("prob", ds.es.prob.shape, ds.es.prob.dtype, float(ds.es.prob[5].min()), float(ds.es.prob[5].max()))
print("dirf", ds.es.dirf.shape, ds.es.dirf.dtype, float(ds.es.dirf[5].min()), float(ds.es.dirf[5].max()))
for m in ds.mm: print("feat", m.shape, m.dtype)
x = ds._x_of(5); print("x", x.shape)
c = pickle.load(open(f"{EV.R}/runs/e94_a/graphs_val_clean.pkl","rb"))
print(c.keys(), len(c["preds"]), c["ks"][:5] if "ks" in c else "")
import inspect, p16_infer as PI
src = inspect.getsource(PI.EvalSet)
print(src[:1500])
