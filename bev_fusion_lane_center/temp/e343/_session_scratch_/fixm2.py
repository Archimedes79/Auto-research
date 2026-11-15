lines = open("mutate.py", encoding="utf-8").read().split("\n")
assert "get_dump no provenance" in lines[34]
lines[34] = ''' ("get_dump no provenance", "    elif expect is not None and (make is not None or os.path.exists(head_path(path))):", "    elif False:"),'''
open("mutate.py", "w", encoding="utf-8").write("\n".join(lines))
