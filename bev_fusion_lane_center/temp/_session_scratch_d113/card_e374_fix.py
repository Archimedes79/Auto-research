import io
p = "experiments/E374_s3-cross-attention-to-the-sd-map/experiment.md"
s = io.open(p, encoding="utf-8").read()
a = "(ratios 4.9 / 4.6 / 3.8 / 3.0 / 2.6; the 3.8 - 5 of the first draft of this note counted the bins' mean)"
b = "(ratios supported / unsupported computed from these columns: 4.9 / 4.6 / 3.8 / 3.0 / 2.6; the coder's summary said 3.8 - 5, the table's own ratios are 2.6 - 4.9)"
assert s.count(a) == 1
io.open(p, "w", encoding="utf-8").write(s.replace(a, b))
