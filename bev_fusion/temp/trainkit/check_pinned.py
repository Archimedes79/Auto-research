"""scratch: a trainkit/ folder next to train.py (a pinned copy) wins over the live autoresearch/trainkit"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "pinned", "code"))
import train
print("trainkit from", os.path.relpath(train.tk_ckpt.__file__, os.path.dirname(os.path.abspath(__file__))))
