import subprocess

head = subprocess.run(["git", "show", "HEAD:bev_fusion/tests/README.md"], capture_output=True, text=True, encoding="utf-8", check=True).stdout
old = "(`autoresearch/evalsuite/api.py`, `runner_api.py`, adapter `evaluation/bev_eval.py`)"
new = "(`autoresearch/evalsuite/api.py`, the project's Runner `runner/runner.py` named in `evaluation/evalsuite.json`, adapter `evaluation/bev_eval.py`)"
assert head.count(old) == 1, head.count(old)
out = head.replace(old, new).replace("Runner.predict + score = run", "Runner.predict + score = run, the runner key of evalsuite.json")
open("bev_fusion/temp/pitch2/readme_tests_mine.md", "w", encoding="utf-8", newline="").write(out)
print("ok")
