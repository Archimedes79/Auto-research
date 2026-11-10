"""Compose a training config from a base file, presets and `--set` overrides, and plan its start (standard library only).

    python -m trainsuite compose BASE.json --preset small --set steps=500 --set model.camera_features=dual --out OUT.json
    python -m trainsuite check job.json    # job.json: {"id": "t15_a", "config": "BASE.json", "preset": ["small"], "card": {"hyp": ..., "pred": ..., "rule": ...}}: the check and the exact run.py command
    python -m trainsuite run job.json      # also starts it (the same call again continues)

The modes of a training are values in the config, not code paths: a preset is a small json file of overrides (`<presets dir>/<name>.json`, deep-merged in the order
given), `--set a.b=c` overrides one value (c is read as json when it parses: `steps=500`, `flip_aug=0.5`, `train_only=["fusion."]`, else it is a string), the base
file stays as it is, the composed file is written to a scratch path. Every change is printed with its old value and its source, a key that the base file did not
have is marked NEW (a typo shows there; the project's check command refuses an unknown key before a GPU is touched). `check` / `run` need a `trainsuite` block in the
project's project.json: {"presets": dir, "out": dir for the composed files, "check": command with {config}, "train": command with {config}, "runs": run folder,
"prepare": [{"name": "gt", "check": command with {config} (exit 0 = present, 1 = missing, any other exit = the check itself failed), "build": command with {config} (resumable), "wait_min": minutes the training may wait for it}]}.
`check` / `run` run the prepare checks; a missing item stops it unless the job says `"prepare": true`: then the build is its own run (`<id>.prep_<name>`, run.py --no-card, so it shows in the schedule and the ledger)
and the training is started with `--after` it, so a failed build blocks the training instead of letting it start without its data.
`"resolve"` (optional): a command (cwd = the project folder) that reads the composed config as json on stdin and writes the resolved one on stdout, for names that the project turns into values
(bev_fusion: `"split": "large"` becomes its scene lists, `training/splits.py`); a non-zero exit refuses the start. The resolved config is what is written, checked, compared with a run's earlier
config and trained, so the run pins the values, not the name.
"""
import copy
import difflib
import json
import math
import os
import shlex
import subprocess

MISSING = "<missing>"


class ComposeError(ValueError):
    pass


class _NonFinite(ValueError):
    pass


def _no_const(name):
    """json parse_constant: NaN / Infinity / -Infinity are no json (the writer would emit invalid json for them)"""
    raise _NonFinite(f"{name} is not a json number")


def _clean(v, what, ints):
    """v with integral floats as ints when `ints` (steps=1e3 is 1000, not 1000.0); ComposeError for a non-finite float (json reads 1e400 as inf)"""
    if isinstance(v, float):
        if not math.isfinite(v):
            raise ComposeError(f"{what}: {v} is not a finite number")
        return int(v) if ints and v.is_integer() else v
    if isinstance(v, dict):
        return {k: _clean(x, what, ints) for k, x in v.items()}
    if isinstance(v, list):
        return [_clean(x, what, ints) for x in v]
    return v


def load(path, ints=False):
    """a json object from a file; `ints`: integral floats become ints (the presets, as --set does it; the base file stays as it is)"""
    try:
        with open(path, encoding="utf-8") as f:
            d = json.load(f, parse_constant=_no_const)
    except OSError as e:
        raise ComposeError(f"cannot read {path}: {e}")
    except ValueError as e:
        raise ComposeError(f"{path} is not json: {e}")
    if not isinstance(d, dict):
        raise ComposeError(f"{path}: the config must be a json object")
    return _clean(d, path, ints)


def parse_set(text):
    """'a.b=c' -> (['a', 'b'], value); the value is json when it parses (500, 0.5, true, null, [..], {..}), else the string; an integral float
    is an int (1e3 -> 1000), NaN / Infinity / 1e400 are refused"""
    key, sep, raw = text.partition("=")
    key = key.strip()
    if not sep or not key or any(not p for p in key.split(".")):
        raise ComposeError(f"--set wants key=value or a.b=value, got {text!r}")
    try:
        value = json.loads(raw, parse_constant=_no_const)
    except _NonFinite as e:
        raise ComposeError(f"--set {text!r}: {e}")
    except ValueError:
        return key.split("."), raw
    return key.split("."), _clean(value, f"--set {text!r}", True)


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
        for path_, v in _flat([], load(path, ints=True)):
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
    """write the composed config atomically (a half file must never be read by a trainer); a NaN / Infinity is refused, never written as invalid json"""
    try:
        text = json.dumps(cfg, indent=1, ensure_ascii=False, allow_nan=False) + "\n"
    except ValueError as e:
        raise ComposeError(f"{out}: the config has a value that is not json ({e})")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    tmp = out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, out)


def resolve_config(cmd, cfg, cwd):
    """(resolved config, notes): cfg passed through the project's resolve command (json on stdin, json on stdout; see the module doc); the notes name every key it set, changed or removed"""
    try:
        r = subprocess.run(shlex.split(cmd), input=json.dumps(cfg), cwd=cwd, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as e:
        raise ComposeError(f"resolve: cannot run {cmd!r}: {e}")
    if r.returncode:
        raise ComposeError(f"resolve ({cmd}) refused: {(r.stderr or r.stdout).strip()[-400:]}")
    try:
        out = _clean(json.loads(r.stdout, parse_constant=_no_const), "resolve", False)
    except ValueError as e:
        raise ComposeError(f"resolve ({cmd}): its output is not json: {e}")
    if not isinstance(out, dict):
        raise ComposeError(f"resolve ({cmd}): its output must be a json object")
    show = lambda v: f"{len(v)} entries" if isinstance(v, list) else json.dumps(v)                 # noqa: E731 - a scene list is not printed
    notes = [f"resolve: {k} = {show(v)}" for k, v in out.items() if cfg.get(k, MISSING) != v] + [f"resolve: {k} removed" for k in cfg if k not in out]
    return out, notes


def same_folder_ids(records, run_id):
    """the OTHER run ids of the run records whose run folder is the same: the folder is the id with '.' as '_', so t1.a and t1_a share runs/t1_a"""
    folder = run_id.replace(".", "_")
    return sorted(r for r in os.listdir(records) if r != run_id and r.replace(".", "_") == folder) if records and os.path.isdir(records) else []


def gpu_gate(block):
    """`gpu_max_mb` of the project's trainsuite block: a training starts only when less than this many MB of the GPU are in use (run.py --gpu-max-mb). 0 when the key is absent or
    0; a value that is no whole positive number (true, "3000", -1, 2.5) is refused: a gate of `true` would be 1 MB and no training would ever start"""
    v = block.get("gpu_max_mb")
    if v is None or (not isinstance(v, bool) and v == 0):
        return 0
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 or v != int(v):
        raise ComposeError(f'"gpu_max_mb" of the trainsuite block must be a whole number of MB (3000), not {v!r}')
    return int(v)


def plan_start(project_root, block, base_path, run_id, presets=(), sets=(), hyp=None, pred=None, rule=None, no_card=False, extra=(), ar_tools=None, records=None):
    """everything `run` will do, without doing it: {config, held (the id is ON HOLD), composed, changes, check (argv list or None), argv (the run.py command), notes, log}.
    Raises ComposeError for what must be fixed first (unknown preset, a missing card field, no train command, another run id with the same run folder)"""
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
    twins = same_folder_ids(records, run_id)
    if twins:
        raise ComposeError(f"--id {run_id}: its run folder runs/{folder} ('.' becomes '_') is also the folder of the run id {twins[0]}, which has a run record"
                           + (" and is ON HOLD" if os.path.exists(os.path.join(records, twins[0], "HOLD")) else "") + ": two ids must not share one folder; use another id")
    key = block.get("name_key", "name")                       # the config key that names the run folder (runs/<name>); "" = the project has none
    if key and any(s.partition("=")[0].strip() == key for s in sets):
        raise ComposeError(f"--set {key}=...: the run folder is runs/{folder}, named by --id, and the launcher script names it after the config file: change --id instead")
    cfg, changes = compose(base, presets, sets, presets_dir)
    resolved = []
    if block.get("resolve"):                                  # names the project turns into values (a split into its scene lists); before the config is compared with a run's earlier one
        cfg, resolved = resolve_config(block["resolve"], cfg, project_root)
    if key:                                                   # directly, not as a --set: an id like 123, 1e3 or true must stay a string, and a base without the key must not fall back to the default name
        changes.append((key, _put(cfg, [key], folder), folder, "--id"))
    out_dir = os.path.join(project_root, block.get("out", "temp/compose"))
    composed, preview = os.path.join(out_dir, folder + ".json"), os.path.join(out_dir, folder + ".preview.json")
    notes = list(resolved)
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
    held = bool(records) and os.path.exists(os.path.join(records, run_id, "HOLD"))
    if held:
        notes.append(f"run id {run_id} is ON HOLD: run.py will refuse it until the person's words clear it (hold.py show)")
    rel = lambda p: os.path.relpath(p, project_root).replace(os.sep, "/")                    # noqa: E731
    sub = lambda cmd, p: shlex.split(cmd.replace("{config}", shlex.quote(rel(p))))           # noqa: E731 - not .format: a ${HOME} in the command is no field; quoted: a space in the folder
    check = sub(block["check"], composed) if block.get("check") else None
    check_preview = sub(block["check"], preview) if block.get("check") else None
    run_py = os.path.join(ar_tools or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"), "run.py")
    argv = ["python3", run_py, "--id", run_id, "--gpu", "--split", "train"]
    if gpu_gate(block):                                          # a training starts only when the card is nearly empty: a second big job next to it would page the GPU memory and slow both
        argv += ["--gpu-max-mb", str(gpu_gate(block))]
    if no_card:
        argv.append("--no-card")
    for flag, val in (("--hyp", hyp), ("--pred", pred), ("--rule", rule)):
        if val:
            argv += [flag, val]
    argv += list(extra) + ["--"] + sub(train, composed)
    prepare = []                                                 # data a run needs before it starts (the GT store of its version): checked, and built as a run of its own when missing
    for item in block.get("prepare") or []:
        name = str(item.get("name") or "")
        if not name or not all(c.isalnum() or c in "_-" for c in name) or not item.get("check") or not item.get("build"):
            raise ComposeError(f'"prepare" entries of the trainsuite block need a name (letters, digits, _ -), a check and a build command: {item!r}')
        pid = f"{run_id}.prep_{name}"
        prepare.append(dict(name=name, id=pid, check=sub(item["check"], composed), check_preview=sub(item["check"], preview), wait_min=int(item.get("wait_min", 1440)),
                            argv=["python3", run_py, "--id", pid, "--no-card", "--"] + sub(item["build"], composed)))
    return dict(config=cfg, held=held, composed=composed, preview=preview, changes=changes, base_keys=keys_of(base), check=check, check_preview=check_preview, argv=argv, notes=notes, prepare=prepare,
                log=os.path.join(project_root, block.get("runs", "runs"), folder + ".launcher.log"))
