"""scratch: replace the TrainConfig block of train.py (from '@dataclass' to the glue banner) by new_trainconfig.txt"""
import sys

path = sys.argv[1]
src = open(path, encoding="utf-8", newline="").read()
new = open(sys.argv[2], encoding="utf-8", newline="").read()
a = src.index("@dataclass\nclass TrainConfig:")
b = src.index("# ------------------------------------------------------------------------------------------------ glue")
if "\r\n" in src:
    new = new.replace("\r\n", "\n").replace("\n", "\r\n")
open(path, "w", encoding="utf-8", newline="").write(src[:a] + new + src[b:])
print("replaced", a, b)
