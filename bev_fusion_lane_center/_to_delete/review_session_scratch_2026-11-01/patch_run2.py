import sys
p, d = sys.argv[1], sys.argv[2]
s = open(p, encoding="utf-8").read()
def rep(old, new, count=1):
    global s
    assert s.count(old) == count, (old[:70], s.count(old))
    s = s.replace(old, new)
rep('''def run_once(cmd, cwd, log_path, hb_path, timeout_s):''', '''def run_once(cmd, cwd, log_path, hb_path, timeout_s, env=None):''')
rep('''        p = subprocess.Popen(cmd, cwd=cwd, stdout=lg, stderr=subprocess.STDOUT, start_new_session=POSIX)''',
    '''        p = subprocess.Popen(cmd, cwd=cwd, stdout=lg, stderr=subprocess.STDOUT, start_new_session=POSIX, env=env)''')
rep('''                rc, out, pid, timed_out = run_once(cmd, a.cwd, os.path.join(d, "run.log"), hb, a.timeout * 60)''',
    '''                rc, out, pid, timed_out = run_once(cmd, a.cwd, os.path.join(d, "run.log"), hb, a.timeout * 60, cmd_env)''')
rep('''        rc, attempt, tail, timed_out = 1, 0, "", False''',
    '''        rc, attempt, tail, timed_out = 1, 0, "", False
        # the command learns which code snapshot it is (a launcher that trains from `git archive $AR_COMMIT` runs exactly the recorded code,
        # also on a resumed id whose tag still names attempt 1)
        cmd_env = dict(os.environ, AR_RUN_ID=a.id, **({"AR_COMMIT": commit, "AR_TAG": tag or prev.get("tag", "")} if commit not in ("skipped", "no-git", "git-error") else {}))''')
rep('''    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()''', '''    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    if cfg.get("gpu_slots_dir") and not os.path.isdir(os.path.dirname(slot_dir(root, cfg, records))):
        print(f"[{a.id}] warning: gpu_slots_dir {cfg['gpu_slots_dir']} lies in a folder that does not exist: the GPU slots are NOT shared with the other problem")''')
rep('''                (relative to the problem folder; problems that share one GPU name the same folder), default <run_records>/.gpu_slots, taken''',
    '''                (relative to the problem folder; problems that share one GPU name the same folder AND the same gpu_slots count), default
                <run_records>/.gpu_slots, taken''')
rep('''  * timeout     --timeout <minutes>:''', '''  * environment the command gets AR_RUN_ID and, with a snapshot, AR_COMMIT / AR_TAG (the commit this attempt ran; the tag names attempt 1)
  * timeout     --timeout <minutes>:''')
open(p, "w", encoding="utf-8", newline="").write(s)
s = open(d, encoding="utf-8").read()
rep('''# TEMPORARY (D24, 2026-10-31): the lane problem's names, used only while its project.json has no "lower_is_better" key (that file is
# protected and the grant is pending); remove this set once the key is there''', '''# TEMPORARY (D24, 2026-10-31): the lane problem's names (tag prefix lcr), used only while its project.json has no "lower_is_better" key
# (that file is protected and the grant is pending); remove this set once the key is there''')
rep('''    return LOWER | set(c["lower_is_better"] if "lower_is_better" in c else LEGACY_LANE_LOWER)''',
    '''    if "lower_is_better" in c:
        return LOWER | set(c["lower_is_better"])
    return LOWER | (LEGACY_LANE_LOWER if c.get("tag_prefix") == "lcr" else set())''')
rep('''    if hc is None and effect > band:  # no min_effect clause on the headline (e.g. only a guard): its evidence is judged here (review 2026-10-31, T1)''',
    '''    if hc is None and effect > band and cards[variant].get(""):  # no min_effect clause on the headline (e.g. only a guard): its evidence is judged here
        # (review 2026-10-31, T1); only for whole-split cards: with subset-only cards the "seeds" of head() are subsets, which a guard's at_least judges''')
rep('''    headline metric, else the problem's project.json "lower_is_better" list, else the generic cost names (LOWER)''',
    '''    headline metric, else the generic cost names (LOWER) plus the problem's project.json "lower_is_better" list''')
rep('''    """(effect, paired diffs, interval or None) of a metric between two lists of (metrics, card, ref); the sign is 'higher is better'"""''',
    '''    """(effect, paired diffs, interval or None) of a metric between two lists of (metrics, card, ref); the sign is 'higher is better'.
    lower: the direction (pass it: the default reads the AR_PROJECT / default problem, not necessarily the experiment's)"""''')
open(d, "w", encoding="utf-8", newline="").write(s)
print("ok")
