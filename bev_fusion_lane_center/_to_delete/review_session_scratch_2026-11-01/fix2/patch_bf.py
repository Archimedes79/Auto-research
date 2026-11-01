import sys
nd, tr = sys.argv[1], sys.argv[2]
def patch(p, pairs):
    s = open(p, encoding="utf-8").read()
    for old, new in pairs:
        assert s.count(old) == 1, (p, old[:70], s.count(old))
        s = s.replace(old, new)
    open(p, "w", encoding="utf-8", newline="").write(s)
patch(nd, [
('''            except Exception:                         # noqa: BLE001  an empty / truncated file of a killed writer: built again
                pass''', '''            except Exception as e:                    # noqa: BLE001  an empty / truncated file of a killed writer: built again
                if not getattr(self, "_told_rebuild", False):     # said once per process: a cache that never reads (permissions) shows
                    print(f"GT cache: {os.path.basename(path)} unreadable ({type(e).__name__}), built again", flush=True)
                    self._told_rebuild = True'''),
])
patch(tr, [
('''    rec = {"gt_signature": sig}
    if os.path.exists(path):''', '''    rec = {"gt_signature": sig}
    if not os.path.exists(path) and any(f.startswith("ckpt_") and f.endswith(".pt") for f in os.listdir(out_dir)):
        rec["earlier"] = ["unknown: the run has checkpoints from before gt.json existed"]   # a resume of a run started before 2026-10-31
    if os.path.exists(path):'''),
])
print("ok")
