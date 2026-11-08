"""Batch 3 of the structure review: a resumed RunStore must be the same run (review finding: a re-evaluation after more training, with
another --ckpt-dir or seed, silently returned the OLD stored counts while metrics.json printed the NEW step)."""
import os

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
path = f"{ROOT}/autoresearch/evalsuite/store.py"
s = open(path, encoding="utf-8").read()
old = '''    def create(cls, path: str, manifest: Dict[str, Any], resume: bool = True) -> "RunStore":
        """a new run directory, or (resume) the existing one: frames already stored are kept."""
        if os.path.exists(os.path.join(path, "manifest.json")):
            if not resume:
                raise FileExistsError(f"{path} already holds a run (resume it or choose another name)")
            st = cls.open(path, mode="a")
'''
new = '''    def create(cls, path: str, manifest: Dict[str, Any], resume: bool = True, force: bool = False) -> "RunStore":
        """a new run directory, or (resume) the existing one: frames already stored are kept. A resume must be the SAME run (state,
        source, subject, task, selection as in the stored manifest, the device aside): otherwise the stored counts would be mixed
        into the numbers of another checkpoint / seed / frame set. force=True skips that check."""
        if os.path.exists(os.path.join(path, "manifest.json")):
            if not resume:
                raise FileExistsError(f"{path} already holds a run (resume it or choose another name)")
            st = cls.open(path, mode="a")
            diff = [] if force else _identity_differences(st.manifest, manifest)
            if diff:
                raise ValueError(f"{path} holds a different run ({'; '.join(diff)}): choose another name, or force=True to resume it anyway")
'''
assert s.count(old) == 1
s = s.replace(old, new)
helper = '''

_IDENTITY_KEYS = ("state", "source", "subject", "task", "selection")
_VOLATILE = {"subject": ("device",)}


def _identity_differences(stored: Dict[str, Any], new: Dict[str, Any]) -> List[str]:
    """the manifest entries that make a resume a different run (what differs, stored -> new)"""
    out = []
    for k in _IDENTITY_KEYS:
        a, b = stored.get(k), new.get(k)
        if isinstance(a, dict) and isinstance(b, dict):
            drop = _VOLATILE.get(k, ())
            a, b = {x: v for x, v in a.items() if x not in drop}, {x: v for x, v in b.items() if x not in drop}
            bad = sorted(x for x in set(a) | set(b) if a.get(x) != b.get(x))
            if bad:
                out.append(f"{k}: " + ", ".join(f"{x} {a.get(x)!r} -> {b.get(x)!r}" for x in bad[:3]))
        elif a != b:
            out.append(f"{k}: {a!r} -> {b!r}")
    return out
'''
anchor = "\n\nclass RunStore:"
assert s.count(anchor) == 1
s = s.replace(anchor, helper + anchor)
open(path, "w", encoding="utf-8", newline="").write(s)
print("edited autoresearch/evalsuite/store.py")
