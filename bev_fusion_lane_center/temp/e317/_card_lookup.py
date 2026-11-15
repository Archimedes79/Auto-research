"""Scratch (read-only): which card does run.py find for each run id of the three cards (experiment.card_for_run), and the arms lock."""
import os, sys
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, os.path.join(ROOT, "..", "autoresearch", "tools"))
os.chdir(ROOT)
import experiment as X
for rid in ("E317.g_s0", "E317.g_s1", "E318.u_s0", "E318.u_s1", "E318.c_s0", "E319.m_s0", "E319.m_s1", "E319.p_s0", "E319.p_s1"):
    r = X.card_for_run(rid)
    print(rid, "->", None if r is None else (r[0]["id"], os.path.basename(os.path.dirname(r[1]))))
