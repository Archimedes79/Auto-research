import os
os.chdir("/mnt/c/Auto-research")
p = "autogt/cli.py"
s = open(p, encoding="utf-8").read()

# 1 a worker task: one group of sequences, its devkit loaded once
anchor = "def check_resume(sig_dir, grid, ss):"
task = '''def _group_task(args):
    """one worker, one group of sequences: its memory-light devkit is loaded once (that takes minutes on a busy box), then its missing
    keyframes are written one after the other -> (label, keyframes written, seconds)"""
    label, dataroot, version, res, ss, options, grp, todo, sig_dir, verbose = args
    t0 = time.time()
    _init(dataroot, version, res, ss, options, grp)
    done = 0
    for i, t in enumerate(todo):
        done += _work((t, sig_dir))
        if verbose and i % 500 == 0:
            print(f"[{label}] {i}/{len(todo)} keyframes, {time.time() - t0:.0f} s", flush=True)
    return label, done, time.time() - t0


'''
assert anchor in s
s = s.replace(anchor, task + anchor, 1)

# 2 generate: groups, one per worker
a = s.index("    done, stored = 0, sets.held(sig_dir)\n    for g0 in range(0, len(names), GROUP_SCENES):")
b = s.index("    write_index(sig_dir, ident, version, names, stride, n_total, options, run=run_manifest(")
new = '''    stored = sets.held(sig_dir)
    size = min(GROUP_SCENES, max(1, -(-len(names) // workers)))           # with several workers every worker gets a group of its own
    groups = []
    for g0 in range(0, len(names), size):
        grp = names[g0:g0 + size]
        todo = [t for n in grp for t in tokens[n] if t not in stored]
        if todo:                                                          # a finished group needs no devkit load
            groups.append((f"group {g0 // size + 1}/{-(-len(names) // size)}", grp, todo))
    done = 0
    if workers <= 1 or len(groups) <= 1:
        for label, grp, todo in groups:
            gen = AutoGT(dataroot, version, ident.grid, ss, light=True, scenes=grp, **options)
            done += _write_tokens(gen, todo, sig_dir, 1, None, verbose)
            del gen
            gc.collect()
    else:                       # one group per worker: the devkit of a group is loaded once, by the worker that writes it, not once per worker
        jobs = [(label, dataroot, version, res, ss, options, grp, todo, sig_dir, verbose) for label, grp, todo in groups]
        with mp.get_context("spawn").Pool(min(workers, len(jobs)), maxtasksperchild=1) as pool:
            for label, n, secs in pool.imap_unordered(_group_task, jobs):
                done += n
                if verbose:
                    print(f"{label} done: {n} keyframes written in {secs:.0f} s", flush=True)
'''
s = s[:a] + new + s[b:]
open(p, "w", encoding="utf-8", newline="\n").write(s)
print("generate: one group per worker")
