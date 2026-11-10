import sys, io, contextlib
sys.path.insert(0, ".")
from evalsuite import apicli
print([n for n in dir(apicli) if not n.startswith("__")])
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    code = apicli.dispatch(["methods"])
print(code, len(buf.getvalue()))
print(buf.getvalue())
