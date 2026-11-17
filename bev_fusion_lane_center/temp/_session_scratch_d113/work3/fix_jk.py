import os
p = os.path.join(os.environ["BFLC_ROOT"], "evaluation/jmap_eval.py")
s = open(p, encoding="utf-8").read()
old = "import jmap_train as JMT  # noqa: E402\nimport jpast_eval as JPE  # noqa: E402\nimport jtoken_eval as JE  # noqa: E402\n"
assert old in s
s = s.replace(old, "import jmap_train as JMT  # noqa: E402\nimport jpast_eval as JPE  # noqa: E402\nimport jtoken as JK  # noqa: E402\nimport jtoken_eval as JE  # noqa: E402\n")
open(p, "w", encoding="utf-8").write(s)
print("ok")
