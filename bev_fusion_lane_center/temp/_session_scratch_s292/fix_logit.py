p = "evaluation/jwire_data.py"
s = open(p, encoding="utf-8").read()
a = "logit=np.array([[3.0], [3.0]]).reshape(1, 1, 1)[0, 0].repeat(1)[None].repeat(2, 0)[:1].reshape(1, 1) if False else np.array([[5.0]]),"
assert s.count(a) == 1
open(p, "w", encoding="utf-8", newline="\n").write(s.replace(a, "logit=np.array([[5.0]]),"))
