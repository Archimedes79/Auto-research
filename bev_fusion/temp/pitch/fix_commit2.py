p = "bev_fusion/temp/pitch/commit_eval2.sh"
t = open(p, encoding="utf-8").read()
a = "  bev_fusion/evaluation/bev_mutants.py bev_fusion/evaluation/bev_suite.py bev_fusion/evaluation/bev_task.py bev_fusion/tests/run_gates.sh\npython3 \"$HERE/readme_blobs2.py\""
b = "  bev_fusion/evaluation/bev_mutants.py bev_fusion/evaluation/bev_suite.py bev_fusion/evaluation/bev_task.py\npython3 \"$HERE/readme_blobs2.py\"\npython3 \"$HERE/gates_blob.py\""
assert t.count(a) == 1
t = t.replace(a, b)
open(p, "w", encoding="utf-8", newline="\n").write(t)
print("ok")
