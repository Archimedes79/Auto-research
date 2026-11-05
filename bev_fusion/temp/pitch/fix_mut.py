p = "temp/pitch/mutants.py"
t = open(p, encoding="utf-8").read()
t = t.replace("import evaluation.bev_curves as C  # noqa: E402\nimport evaluation.bev_response as R  # noqa: E402\n", "import evaluation.bev_response as R  # noqa: E402  (sets the import paths)\nimport evaluation.bev_curves as C  # noqa: E402\n")
open(p, "w", encoding="utf-8", newline="\n").write(t)
