p = "temp/query_probe.py"
s = open(p, encoding="utf-8").read()
s = s.replace("for v in sys.argv[1:] or (\"all\", \"lanes\"):\n    run(v)", "if __name__ == \"__main__\":\n    for v in sys.argv[1:] or (\"all\", \"lanes\"):\n        run(v)")
open(p, "w", encoding="utf-8").write(s)
