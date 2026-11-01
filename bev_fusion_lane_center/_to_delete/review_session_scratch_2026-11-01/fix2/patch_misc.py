import sys
ck, rn = sys.argv[1], sys.argv[2]
def patch(p, pairs):
    s = open(p, encoding="utf-8").read()
    for old, new in pairs:
        assert s.count(old) == 1, (p, old[:70], s.count(old))
        s = s.replace(old, new)
    open(p, "w", encoding="utf-8", newline="").write(s)
patch(ck, [
('''    if isinstance(x, (np.ndarray, np.generic, int, float, bool)) and isinstance(y, (np.ndarray, np.generic, int, float, bool)):''',
 '''    if hasattr(x, "detach") and hasattr(x, "cpu"):                   # a torch tensor (no torch import here): compare its values
        x = x.detach().cpu().numpy()
    if hasattr(y, "detach") and hasattr(y, "cpu"):
        y = y.detach().cpu().numpy()
    if isinstance(x, (np.ndarray, np.generic, int, float, bool)) and isinstance(y, (np.ndarray, np.generic, int, float, bool)):'''),
('''    except Exception:                                                   # noqa: BLE001  (not comparable: not checked)
        return None''', '''    except Exception:                                                   # noqa: BLE001  (not comparable)
        return None if type(x) is type(y) else f"{path}: {type(x).__name__} vs {type(y).__name__}"'''),
])
patch(rn, [
('''        return after[-1].strip()[:200]''', '''        exc = next((x for x in after[1:] if not x.startswith((" ", "\t"))), after[-1])   # the exception line, not a launcher's later print
        return exc.strip()[:200]'''),
])
print("ok")
