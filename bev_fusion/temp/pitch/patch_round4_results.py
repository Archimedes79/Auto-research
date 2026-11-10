def patch(path, pairs):
    s = open(path, encoding="utf-8").read()
    for old, new in pairs:
        assert s.count(old) == 1, (path, s.count(old), old[:110])
        s = s.replace(old, new)
    open(path, "w", encoding="utf-8").write(s)


s = open("evalsuite/results.py", encoding="utf-8").read()
a = s.index("class Locked(RuntimeError):")
b = s.index("class ResultSet:")
new_lock = '''class Locked(RuntimeError):
    """another process is writing this folder"""


@contextlib.contextmanager
def writer_lock(path: str) -> Iterator[None]:
    """one writer per folder: an exclusive `flock` on the file `<path>.lock` (the kernel drops it when the holder dies, so there is no stale lock to take over); `pid host` of the holder is
    written into the file for the message. The file itself stays. Without `fcntl` (Windows Python) there is no lock: run the suite from WSL"""
    if fcntl is None:
        yield
        return
    lock = os.path.normpath(path) + ".lock"
    os.makedirs(os.path.dirname(lock), exist_ok=True)
    fd = os.open(lock, os.O_CREAT | os.O_RDWR)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            holder = os.read(fd, 200).decode("utf-8", "replace").strip() or "another process"
            raise Locked(f"{path} is being written by {holder} ({lock}); wait for it") from None
        os.ftruncate(fd, 0)
        os.write(fd, f"pid {os.getpid()} on {socket.gethostname()}".encode())
        yield
    finally:
        os.close(fd)                                                  # releases the lock


'''
s = s[:a] + new_lock + s[b:]
open("evalsuite/results.py", "w", encoding="utf-8").write(s)

patch("evalsuite/results.py", [
    ("import os\nimport socket\nimport time\nimport uuid\nimport zipfile\n", "import os\nimport re\nimport socket\nimport time\nimport zipfile\nimport zlib\n"),
    ("import numpy as np\n\nfrom .contracts import FrameRef, Prediction\n", "import numpy as np\n\ntry:\n    import fcntl\nexcept ImportError:                                                    # Windows\n    fcntl = None\n\nfrom .contracts import FrameRef, Prediction\n"),
    ('''def _write_atomic(path: str, data: bytes) -> None:
    tmp = f"{path}.tmp.{os.getpid()}"
    try:''', '''def _write_atomic(path: str, data: bytes) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)                      # (package.py clean removes the emptied frames / taps folders of a set it keeps the manifest of)
    tmp = f"{path}.tmp.{os.getpid()}"
    try:'''),
    ("`writer_lock` keeps two writers (two chats asking the same question) out of one folder.", "`writer_lock` (flock) keeps two writers (two chats asking the same question) out of one folder."),
    ('''    def refs(self) -> List[FrameRef]:
        """the stored frames (scene and index; the dataset key is in the file, not in its name), in time order"""
        out = []
        for f in self._listing():
            scene, _, idx = f[:-4].rpartition("_")
            out.append(FrameRef(scene, int(idx)))
        return sorted(out)''', '''    def refs(self) -> List[FrameRef]:
        """the stored frames (scene and index; the dataset key is in the file, not in its name), in time order; a file whose name is not <scene>_<index>.npz is not a frame"""
        out = []
        for f in self._listing():
            m = re.fullmatch(r"(.+)_(\\d+)\\.npz", f)
            if m:
                out.append(FrameRef(m.group(1), int(m.group(2))))
        return sorted(out)'''),
    ('''    def _tap_file(self, ref: FrameRef, name: str) -> str:
        return os.path.join(self.path, "taps", ref.slug, name.replace("/", "_") + ".npy")''', '''    def tap_path(self, ref: FrameRef, name: str) -> str:
        """the file of a tap of a frame (a '/' in the name becomes '_')"""
        return os.path.join(self.path, "taps", ref.slug, name.replace("/", "_") + ".npy")'''),
    ("os.path.exists(self._tap_file(ref, t)) for t in taps", "os.path.exists(self.tap_path(ref, t)) for t in taps"),
    ('''                arrs = [np.lib.format.read_array(io.BytesIO(z.read(f"a{i}.npy")), allow_pickle=False) for i in range(len(z.namelist()) - 1)]''', '''                arrs = [np.lib.format.read_array(io.BytesIO(z.read(f"a{i}.npy")), allow_pickle=False) for i in range(_n_arrays(tree))]'''),
    ('''            arrays = {t: np.load(self._tap_file(ref, t), allow_pickle=False) for t in taps}''', '''            arrays = {t: np.load(self.tap_path(ref, t), allow_pickle=False) for t in taps}'''),
    ("except (OSError, ValueError, KeyError, IndexError, EOFError, zipfile.BadZipFile) as e:", "except (OSError, ValueError, KeyError, IndexError, EOFError, zlib.error, zipfile.BadZipFile) as e:"),
    ('''            os.makedirs(os.path.dirname(self._tap_file(ref, name)), exist_ok=True)
            tmp = self._tap_file(ref, name) + f".tmp.{os.getpid()}"
            with open(tmp, "wb") as f:
                np.save(f, np.asarray(arr))
            os.replace(tmp, self._tap_file(ref, name))''', '''            buf = io.BytesIO()
            np.save(buf, np.asarray(arr))
            _write_atomic(self.tap_path(ref, name), buf.getvalue())'''),
    ('''    def _write_manifest(self) -> None:
        _write_atomic(''', '''    def save(self) -> None:
        """write the manifest now (close does it too; a long run saves it after each scene, so a killed writer keeps what it recorded)"""
        _write_atomic('''),
    ("        self._write_manifest()\n        return rs\n", "        rs.save()\n        return rs\n"),
    ('''            self.manifest["cost"] = cost
        self._write_manifest()''', '''            self.manifest["cost"] = cost
        self.save()'''),
    ('''def result_id(identity: Dict[str, Any]) -> str:''', '''def _n_arrays(tree: Any) -> int:
    """how many arrays a frame's tree refers to (the highest placeholder index + 1)"""
    if isinstance(tree, dict):
        (tag, val), = tree.items()
        if tag in ("$a", "$s"):
            return val + 1
        return max((_n_arrays(x) for kv in val for x in kv), default=0) if tag == "$d" else max((_n_arrays(x) for x in val), default=0) if tag == "$t" else 0
    if isinstance(tree, list):
        return max((_n_arrays(x) for x in tree), default=0)
    return 0


def result_id(identity: Dict[str, Any]) -> str:'''),
])
print("ok")
