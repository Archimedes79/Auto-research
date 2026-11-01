"""Copy the agent definitions (autoresearch/agents/*.md) into the workspace's .claude/agents/ so Claude Code sessions started in the workspace
can spawn them by name (ar-scout, ar-referee, ...). Run again after editing an agent. Standard library only.

    python autoresearch/tools/install_agents.py [--workspace <folder>]   (default: the folder that contains autoresearch/)
    python autoresearch/tools/install_agents.py --check                  exit 1 when an installed copy differs, is missing or is not ours
"""
import argparse
import filecmp
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import project as P  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workspace", default=os.path.dirname(P.AR))
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    src = os.path.join(P.AR, "agents")
    dst = os.path.join(a.workspace, ".claude", "agents")
    ours = sorted(f for f in os.listdir(src) if f.endswith(".md"))
    if a.check:
        there = sorted(f for f in os.listdir(dst) if f.endswith(".md")) if os.path.isdir(dst) else []
        bad = [f"{f}: missing" for f in ours if f not in there] + [f"{f}: differs" for f in ours if f in there and not filecmp.cmp(os.path.join(src, f), os.path.join(dst, f), shallow=False)]
        bad += [f"{f}: installed but not in autoresearch/agents" for f in there if f not in ours]
        print("\n".join(bad) if bad else f"{len(ours)} agents installed and in step")
        return 1 if bad else 0
    os.makedirs(dst, exist_ok=True)
    for f in ours:
        shutil.copy2(os.path.join(src, f), os.path.join(dst, f))
    print(f"{len(ours)} agents copied to {dst}")


if __name__ == "__main__":
    sys.exit(main())
