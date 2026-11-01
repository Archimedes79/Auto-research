"""Compose a training config from a base file, presets and `--set` overrides, and plan its start (standard library only).

    python -m trainsuite compose BASE.json --preset small --set steps=500 --set model.camera_features=dual --out OUT.json
    python -m trainsuite start BASE.json --id t15_a --preset small --hyp "..." --pred "..." --rule "..."          # prints the check and the exact run.py command
    python -m trainsuite start BASE.json --id t15_a --preset small --hyp ... --go                                    # also starts it (the same command again resumes)

The modes of a training are values in the config, not code paths: a preset is a small json file of overrides (`<presets dir>/<name>.json`, deep-merged in the order
given), `--set a.b=c` overrides one value (c is read as json when it parses: `steps=500`, `flip_aug=0.5`, `train_only=["fusion."]`, else it is a string), the base
file stays as it is, the composed file is written to a scratch path. Every change is printed with its old value and its source, a key that the base file did not
have is marked NEW (a typo shows there; the project's check command refuses an unknown key before a GPU is touched). `start` needs a `trainsuite` block in the
project's project.json: {"presets": dir, "out": dir for the composed files, "check": command with {config}, "train": command with {config}, "runs": run folder}.
"""
import copy
import difflib
import json
import os
import shlex

MISSING = "<missing>"


class ComposeError(ValueError):
    pass


def load(path):
    try:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
    except OSError as e:
        raise ComposeError(f"cannot read {path}: {e}")
    except ValueError as e:
        raise ComposeError(f"{path} is not json: {e}")
    if not isinstance(d, dict):
        raise ComposeError(f"{path}: the config must be a json object")
    return d


def parse_set(text):
    """'a.b=c' -> (['a', 'b'], value); the value is json when it parses (500, 0.5, true, null, [..], {..}), else the string"""
    key, sep, raw = text.partition("=")
    key = key.strip()
    if not sep or not key or any(not p for p in key.split(".")):
        raise ComposeError(f"--set wants key=value or a.b=value, got {text!r}")
    try:
        return key.split("."), json.loads(raw)
    except ValueError:
        return key.split("."), raw


def _flat(prefix, d):
    """[(dotted key, value)] of a (nested) override: dicts are walked, everything else is a value"""
    out = []
    for k, v in d.items():
        if isinstance(v, dict) and v:
            out += _flat(prefix + [k], v)
        else:
            out.append((prefix + [k], v))
    return out


def _put(cfg, path, value):
    """set cfg[path] and return the old value (MISSING when the key, or a dict on the way, was not there)"""
    d = cfg
    for p in path[:-1]:
        if not isinstance(d.get(p), dict):
            if p in d:
                raise ComposeError(f"{'.'.join(path)}: {p!r} is a value ({d[p]!r}), not a group of keys")
            d[p] = {}
        d = d[p]
    old = d.get(path[-1], MISSING)
    if isinstance(old, dict):
        if not isinstance(value, dict):
            raise ComposeError(f"{'.'.join(path)} is a group of keys: set one of its keys ({'.'.join(path)}.<key>=...), a value would wipe the group")
        if not value:
            return old                                           # an empty object changes nothing (it must not wipe the group either)
    d[path[-1]] = value
    return old


def compose(base, presets=(), sets=(), presets_dir=None):
    """(config, changes): base (a dict) with the presets merged in the order given, then the --set overrides; changes = [(dotted key, old, new, source)]
    with old MISSING for a key the base did not have. The base dict is not changed"""
    cfg, changes = copy.deepcopy(base), []
    for name in presets:
        if not presets_dir:
            raise ComposeError("--preset needs a presets folder (--presets-dir, or the trainsuite block of project.json)")
        path = os.path.join(presets_dir, name + ".json")
        if not os.path.isfile(path):
            have = sorted(f[:-5] for f in os.listdir(presets_dir) if f.endswith(".json")) if os.path.isdir(presets_dir) else []
            near = difflib.get_close_matches(name, have, 1)
            raise ComposeError(f"no preset {name!r} in {presets_dir}" + (f" (did you mean {near[0]!r}?)" if near else f" (there are: {', '.join(have) or 'none'})"))
        for path_, v in _flat([], load(path)):
            changes.append((".".join(path_), _put(cfg, path_, v), v, f"preset {name}"))
    for text in sets:
        path_, v = parse_set(text)
        for sub, sv in (_flat(path_, v) if isinstance(v, dict) and v else [(path_, v)]):      # an object sets its keys one by one: nothing else in the group is lost
            changes.append((".".join(sub), _put(cfg, sub, sv), sv, "--set"))
    return cfg, changes


def describe(changes, base_keys=None):
    """the changes as lines: `key: old -> new  (source)`; a key the base file did not have is marked NEW, with the nearest base key when there is one (a typo)"""
    lines = []
    for key, old, new, src in changes:
        if old is MISSING or old == MISSING:
            near = difflib.get_close_matches(key, base_keys or [], 1) if base_keys else []
            lines.append(f"  NEW {key} = {json.dumps(new)}  ({src}; not in the base file" + (f", close to {near[0]!r}" if near else "") + ")")
        else:
            lines.append(f"  {key}: {json.dumps(old)} -> {json.dumps(new)}  ({src})" if (type(old), old) != (type(new), new) else f"  {key}: {json.dumps(new)} unchanged  ({src})")
    return lines


def keys_of(d):
    return [".".join(p) for p, _ in _flat([], d)]


def write(cfg, out):
    """write the composed config atomically (a half file must never be read by a trainer)"""
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    tmp = out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=1, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, out)


def plan_start(project_root, block, base_path, run_id, presets=(), sets=(), hyp=None, pred=None, rule=None, no_card=False, extra=(), ar_tools=None, records=None):
    """everything `start` will do, without doing it: {config, composed, changes, check (argv list or None), argv (the run.py command), notes, log}.
    Raises ComposeError for what must be fixed first (unknown preset, a missing card field, no train command)"""
    if not run_id or not run_id[0].isalnum() or not all(c.isalnum() or c in "._-" for c in run_id):
        raise ComposeError(f"--id {run_id!r}: starts with a letter or digit; letters, digits, . _ - only")
    train = block.get("train")
    if not train or "{config}" not in train:
        raise ComposeError('project.json "trainsuite" needs "train": a command with {config}')
    missing = [n for n, v in (("--hyp", hyp), ("--pred", pred), ("--rule", rule)) if not v]
    if missing and not no_card:
        raise ComposeError(f"a run that may feed a verdict needs its card before it starts: {', '.join(missing)} (or --no-card for a diagnostic run)")
    presets_dir = os.path.join(project_root, block["presets"]) if block.get("presets") else None
    base = load(base_path if os.path.isabs(base_path) else os.path.join(project_root, base_path))
    sets, folder = list(sets), run_id.replace(".", "_")
    key = block.get("name_key", "name")                       # the config key that names the run folder (runs/<name>); "" = the project has none
    if key and any(s.partition("=")[0].strip() == key for s in sets):
        raise ComposeError(f"--set {key}=...: the run folder is runs/{folder}, named by --id, and the launcher script names it after the config file: change --id instead")
    cfg, changes = compose(base, presets, sets, presets_dir)
    if key:                                                   # directly, not as a --set: an id like 123, 1e3 or true must stay a string, and a base without the key must not fall back to the default name
        changes.append((key, _put(cfg, [key], folder), folder, "--id"))
    out_dir = os.path.join(project_root, block.get("out", "temp/compose"))
    composed, preview = os.path.join(out_dir, folder + ".json"), os.path.join(out_dir, folder + ".preview.json")
    notes = []
    run_dir = os.path.join(project_root, block.get("runs", "runs"), folder)
    has_run = os.path.isdir(run_dir) and any(f.startswith("ckpt_") or f == "log.jsonl" for f in os.listdir(run_dir))
    if has_run and os.path.isfile(os.path.join(run_dir, ".done")):
        notes.append(f"{os.path.relpath(run_dir, project_root)} is a FINISHED run (.done): run.py refuses the id unless --again")
    elif has_run:
        notes.append(f"{os.path.relpath(run_dir, project_root)} has a run: the same command again RESUMES it from its last checkpoint")
    if has_run and os.path.isfile(composed):
        try:
            same = load(composed) == cfg
        except ComposeError:
            same = False
        if not same:
            raise ComposeError(f"{os.path.relpath(composed, project_root)} is the config of the run in {os.path.relpath(run_dir, project_root)} and differs from this one: a resume must have the "
                               "same config (the launcher script re-reads this file on every retry). Use the same presets / --set, or a new --id")
    if records and os.path.exists(os.path.join(records, run_id, "HOLD")):
        notes.append(f"run id {run_id} is ON HOLD: run.py will refuse it until the person's words clear it (hold.py show)")
    rel = lambda p: os.path.relpath(p, project_root).replace(os.sep, "/")                    # noqa: E731
    sub = lambda cmd, p: shlex.split(cmd.replace("{config}", shlex.quote(rel(p))))           # noqa: E731 - not .format: a ${HOME} in the command is no field; quoted: a space in the folder
    check = sub(block["check"], composed) if block.get("check") else None
    check_preview = sub(block["check"], preview) if block.get("check") else None
    run_py = os.path.join(ar_tools or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"), "run.py")
    argv = ["python3", run_py, "--id", run_id, "--gpu", "--split", "train"]
    if no_card:
        argv.append("--no-card")
    for flag, val in (("--hyp", hyp), ("--pred", pred), ("--rule", rule)):
        if val:
            argv += [flag, val]
    argv += list(extra) + ["--"] + sub(train, composed)
    return dict(config=cfg, composed=composed, preview=preview, changes=changes, base_keys=keys_of(base), check=check, check_preview=check_preview, argv=argv, notes=notes,
                log=os.path.join(project_root, block.get("runs", "runs"), folder + ".launcher.log"))
