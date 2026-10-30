p = "tests/radar_codecs.py"
s = open(p, encoding="utf-8").read()
s = s.replace('''Checks: the codec itself is lossless (decode the raster without learning = the slot ceiling); then the learned probes.
Run from the bev_fusion folder: wsl ~/miniconda3/envs/gcar/bin/python -u tests/radar_codecs.py [steps]''', '''Checks: the codec itself is lossless (decode the raster without learning = the oracle ceiling); then the learned probes.
Levels (second argument): none = D (codecs x capacity C = 64 / 128, ReLU probe); e = E (probe architecture: activation relu / leaky,
default / identity start - the probe starts as a pass-through, as a residual block does); f = F (tolerances x 0.25, two seeds,
leaky + identity start). Results: tests/README.md and plan.md section 3.4.
Run from the bev_fusion folder: wsl ~/miniconda3/envs/gcar/bin/python -u tests/radar_codecs.py [steps] [e | f]''')
open(p, "w", encoding="utf-8").write(s)
