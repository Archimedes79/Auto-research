set -e
cd /mnt/c/Auto-research/bev_fusion_lane_center
python3 - <<'EOF'
p = "training/data/lane_past_data.py"
s = open(p, encoding="utf-8").read()
a = 'assert token_errors(np.array([[1.0, 1.0], [100.0, 100.0], [20.0, 20.0]]), np.array([[1.0, 2.0], [50.0, 50.0]]), np.eye(2), np.zeros(2), "identity").tolist() == [1.0]'
b = 'assert token_errors(np.array([[1.0, 1.0], [100.0, 100.0], [20.0, 20.0]]), np.array([[1.0, 2.0], [100.0, 100.5]]), np.eye(2), np.zeros(2), "identity").tolist() == [1.0]'
assert s.count(a) == 1
s = s.replace(a, b)
open(p, "w", encoding="utf-8").write(s)
print("ok")
EOF
