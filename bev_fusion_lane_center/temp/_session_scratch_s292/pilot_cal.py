import os, pickle, sys
R = os.environ["BFLC_ROOT"]
sys.path.insert(0, R + "/evaluation")
import jwire_data as JD
tr = pickle.load(open(R + "/temp/jwire/train_records_s0_max200.pkl", "rb"))
va = pickle.load(open(R + "/temp/jwire/val_records_s0.pkl", "rb"))
pt, pv = JD.pilot_stats(tr), JD.pilot_stats(va)
print("TRAIN pilot (200 frames):", {k: round(v, 3) for k, v in pt.items() if k != "counts"})
print("VAL (873 frames):        ", {k: round(v, 3) for k, v in pv.items() if k != "counts"})
s, (o, m) = JD.calibrate_jitter(tr, (pv["overall"], pv["multi"]))
print(f"jitter scale {s:.3f} m -> TRAIN + jitter agreement overall {o:.3f} real {m:.3f} (VAL {pv['overall']:.3f} / {pv['multi']:.3f})")
