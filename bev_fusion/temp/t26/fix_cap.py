p = "tests/test_student_window.py"
s = open(p).read()
blk = '''class _Capture:
    """a stdout that keeps what was printed"""
    def __init__(self):
        self.text = ""

    def write(self, s):
        self.text += s

    def flush(self):
        pass


'''
assert s.count(blk) == 1
s = s.replace(blk, "")
anchor = '''tmp = tempfile.mkdtemp(prefix="student_window_"'''
assert s.count(anchor) == 1
s = s.replace(anchor, blk + anchor)
open(p, "w").write(s)
