p = "plan.md"
s = open(p, encoding="utf-8").read()
a = s.index("Why the side loss is needed at all, measured with a proxy task")
b = s.index("Not yet shown: the same with the real heads")
new = ("Why the side loss is needed at all, measured with a proxy task (cell holds a moving detection, |vr| > 1 m/s; `tests/test_radar_encoder.py`, 2000 steps, 8 train / 2 held-out scenes; probe = fresh 1 x 1 decoder with pass-through start trained 1000 steps on the frozen features, a lower bound; oracle = decoding the raster itself, 95.0 %): task alone F1 0.983 and 52.3 % of the detections kept (with a default-start probe: 21.3 %); task + decaying side loss F1 0.983 and 93.0 % kept (the module's own decoder: 94.1 %); side loss alone 94.1 % (own decoder 94.5 %, task F1 0.195). So the side loss costs the proxy task nothing and keeps 93 % instead of 52 % of the cloud; the task alone throws away mostly range, direction, RCS and age (16-35 % of the detections violate those tolerances, against 2-5 % with the side loss) because the proxy task only needs the Doppler. The gate passes (>= 0.9 x oracle, F1 within 0.03). ")
s = s[:a] + new + s[b:]
open(p, "w", encoding="utf-8").write(s)

p = "tests/README.md"
s = open(p, encoding="utf-8").read()
a = s.index("first run (default-start probe = lower bound)")
b = s.index("|\n", a)
new = ("oracle 95.0 %; held-out kept / task F1: side loss alone 94.1 % / 0.194, task alone 52.3 % / 0.983, task + side loss 93.0 % / 0.983 (module's own decoder 94.1 %); with a default-start probe the same models gave 76.4 / 21.3 / 77.3 %; gate PASSES (2026-10-26) ")
s = s[:a] + new + s[b:]
open(p, "w", encoding="utf-8").write(s)
print("ok")
