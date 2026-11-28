"""The rule of the interface and its test: `problems/`, `autoresearch/` and `datasets/` reach the ground truth through `import GtGenerator` (the names of `GtGenerator.__all__`)
or `python -m GtGenerator COMMAND`, and through nothing below it.

    GtGenerator.check_imports(["problems/bev_fusion"])      # -> ["file:line: the source line  (why)", ...]; [] = the rule holds
    python -m GtGenerator selftest --only interface         # the rule on a planted file, the public table against the docs, the readers on a made-up store, the tree

What `check_imports` refuses in a .py file (an AST walk, nothing is imported):
- an import below the package: `from GtGenerator.<module> import ...`, `import GtGenerator.<module>`, `from GtGenerator import <a name that is not public>` (also `*`);
- the package used as anything but `GtGenerator.<public name>` (also under `import GtGenerator as G`): an attribute that is not public (`GtGenerator.lane_gt`, `GtGenerator.__file__`;
  only `__all__` besides the names), and the package as a value (`getattr(GtGenerator, ...)`, `M = GtGenerator`, `return GtGenerator`);
- on a public name: a member of the class or the key that is not part of the interface (`GtGenerator.GtGenerator.versions`, also on a name bound by `from GtGenerator import
  GtGenerator`; the list is `_MEMBERS` in __init__.py), and any dunder attribute (`GtGenerator.load.__globals__`);
- a string that is the name of the package (`import_module("GtGenerator")`, `sys.modules["GtGenerator"]`, `os.path.join(..., "GtGenerator", ...)`), except in an argv right
  after "-m";
- a string that holds a path or a dotted name to a file of the folder (`GtGenerator.lane_gt`, `GtGenerator/versions.csv`, in a shell line too). A string that is a statement
  of its own (a docstring) is a text and is not looked at; a comment never is.
Allowed: `import GtGenerator [as G]`, `from GtGenerator import <public>`, `GtGenerator.<public>...`, the argv `"-m", "GtGenerator"`, the launcher `GtGenerator/run_nice.sh`.
What it cannot see: an object (`gen = GtGenerator.GtGenerator(...); gen.clean()`), a name put together at run time. A file that does not parse is reported, never skipped.
A file that binds one name both ways (`import GtGenerator` and `from GtGenerator import GtGenerator`) gets the members of the class reported: give the class another name there.
"""
import ast
import contextlib
import importlib
import os
import re

from .where import REPO

PACKAGE = "GtGenerator"
SCANNED = ("problems", "autoresearch", "datasets")            # the folders of the repository the rule binds

# The lines of the real tree that still reach below the interface: file (from the repository root) -> its source lines; the comment names the public replacement. They move in step S2
# of the interface plan (the files belong to the sessions of the two problems). `selftest --only interface` fails for a use that is not listed AND for a line that is no longer
# there: the list can only shrink, and it is empty when S2 is done. For S2: no module is handed on as a value either (the scan refuses `return GtGenerator` and cannot see
# `self.lg.held(...)`): lane_gt_data.py and lane_tok_data.py call GtGenerator.lane_tokens() / GtGenerator.load_lane(token) directly, and `self.lg` goes.
PENDING = {
    "problems/bev_fusion/evaluation/bev_adapter.py": ("from GtGenerator.versions import version_of_key",),                        # from GtGenerator import version_of_key
    "problems/bev_fusion/training/train.py": ("from GtGenerator.versions import version_of_key",),                                # the same
    "problems/bev_fusion/net/model/layers.py": ("from GtGenerator.layers import AREAS, DIST_MAX, LINES, OBJECTS",),               # from GtGenerator import AREAS, DIST_MAX, LINES, OBJECTS
    "problems/bev_fusion/net/prep/nuscenes_dataset.py": ("from GtGenerator.nuscenes_adapter import pose_matrix",                  # from GtGenerator import pose_matrix
                                                         "from GtGenerator.nuscenes_gt import NuScenesGT"),                       # GtGenerator.GtGenerator(dataroot, version, nusc=nusc); cache names: .store_signature()
    "problems/bev_fusion/net/prep/precompute.py": ("from GtGenerator import coverage",),                                          # from GtGenerator import radar_cells
    "problems/bev_fusion/tests/test_nuscenes_dataset.py": ("from GtGenerator.nuscenes_adapter import pose_matrix",),              # from GtGenerator import pose_matrix
    "problems/bev_fusion/tests/support/synthetic_world.py": ("from GtGenerator.layers import DIST_MAX, cell_centres, occlusion",),   # from GtGenerator import DIST_MAX, cell_centres, occlusion
    "problems/bev_fusion/tests/test_splits.py": (                                                                                 # versions = set(GtGenerator.version_names()); the message names no file
        'with open(os.path.join("..", "..", "GtGenerator", "versions.csv"), newline="", encoding="utf-8") as f:',
        '''check(all(d["gt"] in versions for d in (small, large, xl)), f"every split names a GtGenerator version ({large['gt']}) of GtGenerator/versions.csv")'''),
    "problems/bev_fusion/tests/test_precompute.py": (                                                                             # two messages name the module: "GtGenerator.radar_cells"
        'assert torch.equal(out["vis_radar"][0, 0], torch.from_numpy(_radar_cells()(gx, gy, rig.numpy())).float()), "radar fan != GtGenerator.coverage"',
        'print("precompute selftest passed: camera map = lift_pd.camera_visibility, radar fan = GtGenerator.coverage.radar_cells of the calibrated poses, fusion area = camera x radar, "'),
    "problems/lane_center/evaluation/qgraph_eval.py": ("from GtGenerator import lane_gt",),                                       # twice: load_lane(token); lane_tokens() (the file is protected and in the evaluator hash)
    "problems/lane_center/net/prep/lane_gt_data.py": ("from GtGenerator import lane_gt",),                                        # lane_signature(), lane_tokens(), load_lane(token), called directly
    "problems/lane_center/net/prep/foundation/gcar_features.py": ("from GtGenerator.light_nusc import light_nuscenes",),          # from GtGenerator import light_nuscenes
    "problems/lane_center/tools/viz/qgraph_view.py": ("from GtGenerator import lane_gt",                                          # load_lane(token)
                                                      "from GtGenerator import lane_graph_view as V",                             # lane_camera_image(...)
                                                      "from GtGenerator.nuscenes_adapter import pose_matrix"),                    # from GtGenerator import pose_matrix
}

# The lines of the research core that name the folder as a whole and read nothing of it: file -> its source lines. Not violations (check_imports leaves them out); a line
# that changes has to be looked at again, so the test fails for one that is no longer there.
NAMED_ONLY = {
    "autoresearch/research/evaluate.py": ('''SHARED = ("Evaluation", "trainkit", "research", "GtGenerator")     # the packages a run imports besides its problem's own''',),          # a list of package names
    "autoresearch/research/launch.py": ('''SNAPSHOT_PATHS = ("autoresearch", "GtGenerator", "datasets")       # besides the problem's own folder: the code a run may import''',),     # the folder is copied into a run's snapshot
    "datasets/nuscenes/tools/make_manifest.py": ('LINKS = {"gt": {"registry": "GtGenerator/versions.csv", "tags": "autogt/v<N>", "store": "<data root>/groundtruth/<signature>/index.json", '
                                                 '"check": "python -m GtGenerator data <dataroot>"},',),                                                                               # a link written into the manifest
}


def _interface():
    """(the public names, the members a public class offers, the pattern of a path or a dotted name that leads to a module or a table of the folder)"""
    pkg = importlib.import_module(PACKAGE)
    listing = os.listdir(os.path.dirname(os.path.abspath(__file__)))
    files = {f for f in listing if f.endswith((".py", ".csv", ".json"))} | {f[:-3] for f in listing if f.endswith(".py")}
    pattern = re.compile(rf"{PACKAGE}[./\\](?:{'|'.join(sorted(map(re.escape, files), key=len, reverse=True))})(?!\w)")
    return set(pkg.__all__), pkg._MEMBERS, pattern


def _found(tree, public, members, files):
    """(line number, why) of every use below the interface in a parsed file"""
    nodes = list(ast.walk(tree))
    package = {a.asname or PACKAGE for n in nodes if isinstance(n, ast.Import) for a in n.names                       # the names `import GtGenerator [as G]` binds
               if a.name == PACKAGE or (a.name.startswith(PACKAGE + ".") and not a.asname)}
    bound = {a.asname or a.name: a.name for n in nodes if isinstance(n, ast.ImportFrom) and not n.level and n.module == PACKAGE
             for a in n.names if a.name in public and (a.asname or a.name) not in package}                           # local name -> the public name it is
    texts = {id(n.value) for n in nodes if isinstance(n, ast.Expr)}                                                   # a string as a statement: a docstring
    argv = {id(e) for n in nodes if isinstance(n, (ast.List, ast.Tuple)) for flag, e in zip(n.elts, n.elts[1:]) if isinstance(flag, ast.Constant) and flag.value == "-m"}
    owners = {id(n.value) for n in nodes if isinstance(n, ast.Attribute)}                                             # the nodes an attribute is taken from

    def member(owner, node):
        """a member of a public name that is not part of the interface: outside the list of its class, or a dunder"""
        if node.attr.startswith("__") or node.attr not in members.get(owner, (node.attr,)):
            yield node.lineno, f"{PACKAGE}.{owner}.{node.attr} is not part of the interface"
    for node in nodes:
        if isinstance(node, ast.ImportFrom) and not node.level and node.module:
            if node.module.startswith(PACKAGE + "."):
                yield node.lineno, f"imports the module {node.module}"
            elif node.module == PACKAGE:
                for a in node.names:
                    if a.name not in public:
                        yield node.lineno, f"{a.name} is not a public name"
        elif isinstance(node, ast.Import):
            for a in node.names:
                if a.name.startswith(PACKAGE + "."):
                    yield node.lineno, f"imports the module {a.name}"
        elif isinstance(node, ast.Attribute):
            of = node.value
            if isinstance(of, ast.Name) and of.id in package:
                if node.attr not in public and node.attr != "__all__":
                    yield node.lineno, f"{PACKAGE}.{node.attr} is not a public name"
            elif isinstance(of, ast.Name) and of.id in bound:
                yield from member(bound[of.id], node)
            elif isinstance(of, ast.Attribute) and isinstance(of.value, ast.Name) and of.value.id in package and of.attr in public:
                yield from member(of.attr, node)
        elif isinstance(node, ast.Name) and node.id in package and isinstance(node.ctx, ast.Load) and id(node) not in owners:
            yield node.lineno, "the package as a value"
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in texts:
            if node.value.strip("/\\") == PACKAGE:
                if id(node) not in argv:
                    yield node.lineno, "the package named in a string"
            else:
                for m in files.finditer(node.value):
                    yield node.lineno, f"names the file {m.group(0)}"


def _files(paths):
    for path in paths:
        if os.path.isfile(path):
            yield path
            continue
        if not os.path.isdir(path):
            raise ValueError(f"check_imports: {path} is neither a folder nor a file (an empty answer would read as: the rule holds)")
        for folder, dirs, names in os.walk(path):
            dirs[:] = sorted(d for d in dirs if d != "__pycache__" and not d.startswith("."))
            for name in sorted(names):
                if name.endswith(".py"):
                    yield os.path.join(folder, name)


def scan(paths):
    """[(file, line number, source line, why)] of every use below the interface in the .py files of `paths` (folders are walked), in the order of the files and lines"""
    interface, out = _interface(), []
    for path in _files(paths):
        with open(path, "rb") as f:
            raw = f.read()
        try:
            tree = ast.parse(raw)
        except (SyntaxError, ValueError) as e:
            out.append((path, getattr(e, "lineno", None) or 0, "", f"not parsed, so not checked: {type(e).__name__}"))
            continue
        lines = raw.decode("utf-8", "replace").splitlines()
        for line, why in sorted(set(_found(tree, *interface))):
            out.append((path, line, lines[line - 1].strip(), why))
    return out


def _in_repo(path):
    """the path from the repository root with forward slashes: the form of the keys of PENDING and NAMED_ONLY"""
    return os.path.relpath(os.path.abspath(path), REPO).replace(os.sep, "/")


def check_imports(paths):
    """the uses of GtGenerator below its interface in the .py files of `paths` (folders, walked, or files; one path or a list): a list of "file:line: the source line  (why)",
    [] when the rule holds. A test of a problem: `assert not GtGenerator.check_imports([<the problem's folder>])`"""
    return [f"{path}:{line}: {source}  ({why})" for path, line, source, why in scan([paths] if isinstance(paths, str) else paths)
            if source not in NAMED_ONLY.get(_in_repo(path), ())]


# every line marked `# refused: why[; why]` is reported with exactly these reasons, and no other line: one line per form of going below the interface, among the allowed forms
PLANTED = r'''"""GtGenerator.coverage.radar_cells and GtGenerator/lane_gt.py are named in a text: a text is no import"""
import importlib, os, runpy, subprocess, sys
import GtGenerator
import GtGenerator as G
from GtGenerator import load, GtKey, GtGenerator as Gen
import GtGenerator.lane_gt as L                                    # refused: imports the module GtGenerator.lane_gt
from GtGenerator.layers import AREAS                               # refused: imports the module GtGenerator.layers
from GtGenerator import lane_gt, pose_matrix                       # refused: lane_gt is not a public name
from GtGenerator import *                                          # refused: * is not a public name

def allowed(token):
    "GtGenerator/lane_gt.py in the text of a function"
    subprocess.run([sys.executable, "-m", "GtGenerator", "versions"])
    subprocess.run("bash GtGenerator/run_nice.sh -m GtGenerator have --set full", shell=True)
    job = "GtGenerator.GtGenerator.generate_bev_gt", "GtGenerator.store_root"
    key = GtGenerator.GtGenerator.gt_key("bev_fusion"), Gen.generate_lane_gt, G.GtKey.complete, Gen("~/data/nuscenes", "v1.0-mini").keyframes
    return GtGenerator.load(token), G.current_signature(), GtGenerator.__all__, GtGenerator.AREAS.index("walkway"), key, job

def refused(token):
    a = G.versions.table()                                         # refused: GtGenerator.versions is not a public name
    b = getattr(GtGenerator, "lane_gt")                            # refused: the package as a value
    M = GtGenerator                                                # refused: the package as a value
    c = importlib.import_module(".lane_gt", "GtGenerator")         # refused: the package named in a string
    d = importlib.import_module("GtGenerator").where               # refused: the package named in a string
    e = __import__("GtGenerator", fromlist=["lane_gt"]).lane_gt    # refused: the package named in a string
    f = sys.modules["GtGenerator"].versions                        # refused: the package named in a string
    g = subprocess.run("python -m GtGenerator.lane_graph_view x", shell=True)      # refused: names the file GtGenerator.lane_graph_view
    h = runpy.run_path("GtGenerator/lane_gt.py"), open(f"{token}/GtGenerator\\versions.csv")     # refused: names the file GtGenerator/lane_gt.py; names the file GtGenerator\versions.csv
    i = GtGenerator.GtGenerator.versions(update=True), Gen.clean(delete=True)      # refused: GtGenerator.GtGenerator.versions is not part of the interface; GtGenerator.GtGenerator.clean is not part of the interface
    j = GtGenerator.load.__globals__, GtGenerator.__file__         # refused: GtGenerator.load.__globals__ is not part of the interface; GtGenerator.__file__ is not a public name
    k = open(os.path.join("..", "GtGenerator", "versions.csv"))    # refused: the package named in a string
    m = open(os.path.join("..", "GtGenerator/") + "lane_gt" + ".py")       # refused: the package named in a string
    return G                                                       # refused: the package as a value
'''
PLANTED_FOUND = sorted((number, why) for number, line in enumerate(PLANTED.splitlines(), 1) if "# refused: " in line for why in line.split("# refused: ")[1].split("; "))


@contextlib.contextmanager
def _environment(**values):
    """environment variables set (None: taken out) for the length of a with block"""
    before = {k: os.environ.get(k) for k in values}
    try:
        for k, v in values.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v
        yield
    finally:
        for k, v in before.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v


def _write(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _raises(error, call, *args):
    """the message of the `error` that `call(*args)` must raise"""
    try:
        call(*args)
    except error as e:
        return str(e)
    raise AssertionError(f"{getattr(call, '__name__', call)}{args} must raise {error.__name__}")


def _rule_check():
    """the planted file: exactly its marked lines are reported, each with file, line and reason, and none of the allowed forms"""
    import tempfile
    assert len(PLANTED_FOUND) == 21 and len({line for line, _ in PLANTED_FOUND}) == 18, "the planted file holds 18 refused lines with 21 reasons"
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "pkg", "__pycache__"))
        path = os.path.join(d, "pkg", "planted.py")
        _write(path, PLANTED)
        _write(os.path.join(d, "pkg", "__pycache__", "old.py"), "import GtGenerator.store\n")                 # no source: not walked
        _write(os.path.join(d, "pkg", "notes.txt"), "import GtGenerator.store\n")
        got = scan([d])
        assert [(p, line, why) for p, line, _, why in got] == [(path, line, why) for line, why in PLANTED_FOUND], got
        first = check_imports(d)[0]
        assert first.startswith(f"{path}:6: import GtGenerator.lane_gt as L") and first.endswith("  (imports the module GtGenerator.lane_gt)"), first
        assert check_imports([path]) == check_imports(d), "one path or a list, a folder or a file"
        documented = "from GtGenerator import GtGenerator\nkey = GtGenerator.generate_bev_gt('~/data/nuscenes')\n"       # the name is the class here: its listed members are allowed
        _write(path, documented)
        assert scan([d]) == [], "the documented class form is allowed"
        _write(path, documented + "def broken(:\n")
        assert [(line, why[:10]) for _, line, _, why in scan([d])] == [(3, "not parsed")], "a file that does not parse is reported, not skipped"
        _raises(ValueError, check_imports, [os.path.join(d, "nothing")])


def _table_check():
    """the public table is the documented one: `__all__` is the table, every name resolves to its owner and is no module of the folder, the members of a public class exist,
    the package docstring has a contract line for each name and names each member, AGENTS.md section 1 names each as `GtGenerator.<name>`, no line of AGENTS.md sends its reader
    to a `GtGenerator.<name>` that is not public, and the table of README.md lists exactly these names"""
    pkg = importlib.import_module(PACKAGE)
    here = os.path.dirname(os.path.abspath(__file__))
    assert pkg.__all__ == sorted(pkg._PUBLIC), "__all__ is the table"
    modules = {f[:-3] for f in os.listdir(here) if f.endswith(".py")}
    for name in pkg.__all__:
        module, attribute = pkg._PUBLIC[name]
        assert name not in modules and module in modules, f"{name}: a public name is no module of the folder, and its owner {module}.py is one"
        assert getattr(pkg, name) is getattr(importlib.import_module(f"{PACKAGE}.{module}"), attribute), name
    for name, members in pkg._MEMBERS.items():
        owner = getattr(pkg, name)
        gone = [m for m in members if not hasattr(owner, m) and m not in getattr(owner, "__dataclass_fields__", ())]
        assert not gone, f"_MEMBERS: {name} has no {gone}"
        assert not [m for m in members if not re.search(rf"\b{m}\b", pkg.__doc__)], f"the package docstring does not name every member of {name}"
    with open(os.path.join(here, "AGENTS.md"), encoding="utf-8") as f:
        guide = f.read()
    section = guide.split("\n## 1. ")[1].split("\n## ")[0]
    no_contract = [n for n in pkg.__all__ if not re.search(rf"^    (?:\w+, )*{n}\b", pkg.__doc__, re.M)]
    not_named = [n for n in pkg.__all__ if not re.search(rf"`{PACKAGE}\.{n}\b", section)]
    below = sorted({n for n in re.findall(rf"\b{PACKAGE}\.([A-Za-z_]\w*)", guide) if n not in pkg.__all__ and not n.startswith("__")})
    assert not no_contract, f"the package docstring has no contract line for {no_contract}"
    assert not not_named, f"AGENTS.md section 1 does not name {not_named} (as `{PACKAGE}.<name>`)"
    assert not below, f"AGENTS.md names {PACKAGE}.{below}: not public"
    with open(os.path.join(here, "README.md"), encoding="utf-8") as f:
        rows = [line for line in f.read().split("\n## The interface\n")[1].split("\n## ")[0].splitlines() if line.startswith("|")]
    listed = re.findall(r"`([A-Za-z_]\w*)`", "\n".join(rows))
    assert sorted(listed) == pkg.__all__, f"the table of README.md (The interface) and __all__ differ: {sorted(set(listed) ^ set(pkg.__all__))}"
    return len(pkg.__all__)


def _root_check():
    """where the stores are: RESEARCH_DATA, else `data` of the machine file, both read at the call. A machine file that is named or exists and does not give the data root is an
    error that names the file, for the readers too (an empty answer would read as: nothing is stored, generate it); only a machine without any machine file gets the folder
    in the repository"""
    import json
    import tempfile
    pkg = importlib.import_module(PACKAGE)
    with tempfile.TemporaryDirectory() as d:
        good, corrupt, no_data, mistyped = (os.path.join(d, name) for name in ("good.json", "corrupt.json", "no_data.json", "mistyped.json"))
        _write(good, json.dumps({"data": os.path.join(d, "data")}))
        _write(corrupt, '{"data": ')
        _write(no_data, json.dumps({"runs": d}))
        with _environment(RESEARCH_DATA=None, RESEARCH_MACHINE=good):
            assert pkg.store_root() == os.path.join(d, "data", "groundtruth"), "the data root of the machine file that is named"
        for bad in (corrupt, no_data, mistyped):
            with _environment(RESEARCH_DATA=None, RESEARCH_MACHINE=bad):
                assert bad in _raises(ValueError, pkg.store_root) and bad in _raises(ValueError, pkg.lane_tokens), "the error names the file"
            with _environment(RESEARCH_DATA=d, RESEARCH_MACHINE=bad):
                assert pkg.store_root() == os.path.join(d, "groundtruth"), "RESEARCH_DATA decides, the machine file is not read"
        with _environment(RESEARCH_DATA=None, RESEARCH_MACHINE=None, HOME=d):                       # no file is named and none is at the default place: a fresh clone
            assert pkg.store_root() == os.path.join(REPO, "data", "autogt")
            os.makedirs(os.path.join(d, ".research"))
            _write(os.path.join(d, ".research", "machine.json"), '{"data": ')                        # a file at the default place that cannot be read is no fresh clone
            assert os.path.join(d, ".research", "machine.json") in _raises(ValueError, pkg.store_root)


def _names_check():
    """the readers and the identities of the interface on a made-up store root (no nuScenes): the root is read at every call, a lane graph and the stored tokens come from
    <root>/lane/<lane_signature(rig)>/, a generator names its STORE (after same_gt_as) and takes its options by keyword only, the version names are those of the table, and
    neither `import GtGenerator` nor a read of the lane store loads the front generator"""
    import gzip
    import pickle
    import subprocess
    import sys
    import tempfile
    import types
    from . import gt_api, lane_gt, versions
    from .suite import GtGenerator
    pkg = importlib.import_module(PACKAGE)
    token = "0" * 32
    with tempfile.TemporaryDirectory() as d, _environment(RESEARCH_DATA=d):
        root = os.path.join(d, "groundtruth")
        assert pkg.store_root() == root, "the store root follows RESEARCH_DATA at the call, not at the import"
        assert pkg.missing([token]) == [token] and pkg.lane_tokens() == set(), "an empty root holds nothing"
        assert pkg.gt_key("lane_center", deep=False).out == root and not pkg.gt_key("lane_center", deep=False).complete, "the key looks below the root of the call"
        assert gt_api._roots(d, None, "v1.0-mini")[2] == root, "generate_bev_gt / generate_lane_gt write below the root of the call"
        for read in (pkg.load, pkg.load_lane):
            assert root in _raises(FileNotFoundError, read, token), f"{read.__name__} reads below the root of the call"
        for rig in sorted(lane_gt.RIGS):
            folder = os.path.join(root, "lane", pkg.lane_signature(rig))
            os.makedirs(folder)
            with gzip.open(os.path.join(folder, token + ".pkl.gz"), "wb") as f:
                pickle.dump({"rig": rig}, f)
            assert pkg.lane_signature(rig) == lane_gt.signature(rig=rig), "the store key (after same_gt_as), not the code signature"
            assert pkg.lane_tokens(rig) == pkg.lane_tokens(rig, data=root) == {token} and pkg.load_lane(token, rig) == pkg.load_lane(token, rig, root) == {"rig": rig}, rig
        assert len({pkg.lane_signature(rig) for rig in lane_gt.RIGS}) == len(lane_gt.RIGS), "one store per rig"
        quiet = ("import sys, GtGenerator as G; G.store_root(); G.version_of_key('x'); assert 'numpy' not in sys.modules, 'import GtGenerator loads numpy'; "
                 f"assert G.lane_tokens() == {{'{token}'}}; front = [m for m in ('GtGenerator.suite', 'GtGenerator.nuscenes_gt') if m in sys.modules]; assert not front, front")
        r = subprocess.run([sys.executable, "-c", quiet], capture_output=True, text=True, cwd=REPO)
        assert r.returncode == 0, "a fresh process: " + r.stderr[-500:]
    # the store of a generator: new code that makes the GT of an older version (a row with same_gt_as) names that version's store, code without a row its own
    rows = versions.table()
    old = versions.canonical(rows[-1]["signature"])
    real, versions.table = versions.table, lambda: rows + [dict(rows[-1], version="v999", signature="f" * 10, same_gt_as=rows[-1]["version"])]
    try:
        assert GtGenerator.store_signature(types.SimpleNamespace(signature=lambda: "f" * 10)) == old != "f" * 10, "same_gt_as: the store of the older version"
        assert GtGenerator.store_signature(types.SimpleNamespace(signature=lambda: "e" * 10)) == "e" * 10, "no row: its own store"
        assert pkg.version_of_key("0.5_4_" + "f" * 10) == versions.version_of(old) and pkg.version_of_key("0.5_4_" + "e" * 10) is None
        assert pkg.version_names() == [r["version"] for r in rows] + ["v999"] and pkg.version_names()[0] == "v1", "the version names of the table, oldest first"
    finally:
        versions.table = real
    # a third positional argument was the grid of this class and the devkit of NuScenesGT: refused before anything is loaded, not guessed
    assert "positional" in _raises(TypeError, GtGenerator, "~/no_such_folder", "v1.0-mini", object())


def _tree_check():
    """the real tree: every use below the interface is a line of PENDING (to be migrated) or of NAMED_ONLY (the folder by name), and every line of both is still there"""
    roots = [os.path.join(REPO, d) for d in SCANNED if os.path.isdir(os.path.join(REPO, d))]
    got = [(_in_repo(path), line, source, why) for path, line, source, why in scan(roots)]
    new = [f"{p}:{line}: {source}  ({why})" for p, line, source, why in got if source not in PENDING.get(p, ()) + NAMED_ONLY.get(p, ())]
    assert not new, (f"{len(new)} use(s) of {PACKAGE} below its interface (allowed: `import {PACKAGE}` and the names of {PACKAGE}.__all__; a name that is missing there is added "
                     "to the table, not imported from below):\n" + "\n".join(new))
    seen = {(p, source) for p, _, source, _ in got}
    stale = [f"{p}: {source}" for listed in (PENDING, NAMED_ONLY) for p, sources in listed.items() for source in sources if (p, source) not in seen]
    assert not stale, "these lines of interface_selftest.PENDING / NAMED_ONLY are no longer in the tree: take them out of the list\n" + "\n".join(stale)
    assert not check_imports([r for r in roots if os.path.basename(r) != "problems"]), "the research core and datasets/ pass check_imports (a line of NAMED_ONLY is no violation)"
    return len(roots), sum(source in PENDING.get(p, ()) for p, _, source, _ in got)


def run(verbose=True, tree=True):
    """`tree=False` leaves the scan of problems/, autoresearch/ and datasets/ out: the full self-test of the generator (the gate of a signature step) must not fail for a
    half-edited file of another session; `selftest --only interface` and the tests of the problems scan the tree"""
    _rule_check()
    names = _table_check()
    _root_check()
    _names_check()
    msg = (f"interface selftest ok: the planted file gives exactly its {len(PLANTED_FOUND)} violations, {names} public names resolve and are documented, the store root is read at the "
           "call and a machine file that does not give it is an error, a lane read loads no front generator, a generator names its store")
    if tree:
        roots, pending = _tree_check()
        msg += f"; {roots} folder(s) of the tree reach {PACKAGE} only through the interface" + (f" except the {pending} pending line(s) of {len(PENDING)} file(s)" if pending else "")
    if verbose:
        print(msg)
    return msg


if __name__ == "__main__":
    run()
