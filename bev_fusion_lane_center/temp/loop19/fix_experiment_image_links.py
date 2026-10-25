"""One-off (ROADMAP P7): repair image links in the experiment folders after the migration renamed the images (architecture.png -> E39_architecture.png,
rung pictures moved to figures/). A link is repaired only when exactly one file of the experiment folder ends with the linked name.

    wsl python3 temp/loop19/fix_experiment_image_links.py          # dry run: what would change, what stays unresolved
    wsl python3 temp/loop19/fix_experiment_image_links.py --write
"""
import os
import re
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "experiments")
LINK = re.compile(r"(\]\(|`)([^)`\s]+\.(?:png|jpg|svg|gif))([)`])")
write = "--write" in sys.argv
fixed = unresolved = 0
for exp in sorted(os.listdir(ROOT)):
    top = os.path.join(ROOT, exp)
    if not os.path.isdir(top):
        continue
    files = [os.path.join(dp, f) for dp, _, fn in os.walk(top) for f in fn]
    for md in [f for f in files if f.endswith(".md")]:
        here = os.path.dirname(md)
        text = open(md, encoding="utf-8").read()

        def repl(m):
            global fixed, unresolved
            target = m.group(2)
            if target.startswith("http") or os.path.exists(os.path.join(here, target)):
                return m.group(0)
            base = os.path.basename(target)
            hits = [f for f in files if os.path.basename(f) == base or os.path.basename(f).endswith("_" + base)]
            if len(hits) != 1:
                unresolved += 1
                print(f"  unresolved {os.path.relpath(md, ROOT)}: {target} ({len(hits)} candidates)")
                return m.group(0)
            new = os.path.relpath(hits[0], here).replace("\\", "/")
            fixed += 1
            print(f"  {os.path.relpath(md, ROOT)}: {target} -> {new}")
            return m.group(1) + new + m.group(3)

        out = LINK.sub(repl, text)
        if write and out != text:
            open(md, "w", encoding="utf-8", newline="").write(out)
print(f"{fixed} links {'repaired' if write else 'to repair'}, {unresolved} unresolved")
