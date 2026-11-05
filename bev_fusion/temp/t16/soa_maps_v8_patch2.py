"""throw-away: docstring edits of soa_maps_v8.py"""
p = "temp/t16/soa_maps_v8.py"
s = open(p, encoding="utf-8").read()


def rep(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old[:70])
    s = s.replace(old, new)


rep("        [--head] [--stored] [--no-model] [--dt0] [--prior N] [--offset M] [--align] [--no-check] [--store DIR]",
    "        [--head] [--stored] [--no-model] [--all-dts] [--rule old|usable|old400] [--prior N] [--offset M] [--align] [--no-check] [--store DIR]")
rep('''  LIMIT      number of frames, default 400 = the 400 frames of the old scoring (temp/soa_gap/map_ref_dev.py): the GaussianCaR logits cached by the
             lane project (val_tokens.json, 1706 frames in 43 of our 82 val scenes) that are samples of our val set, every 4th of them
             (rows[::len(rows)//LIMIT][:LIMIT]). A smaller LIMIT spreads over the same list with a larger step (a quick look).
  --head     the FIRST LIMIT frames of the 400 instead of re-spreading (a LIMIT run is then a prefix of the full run)
  --stored   only frames of the 400 that are also in the run's stored 1000-frame suite evaluation (sanity gate: the per-frame counts of our model are
             compared with the stored ones, which must be equal); the check runs for every scored frame that is in that store anyway
  --no-model GaussianCaR and the prior only (no network, a few minutes for 400 frames): the check of the reference against the old numbers
  --dt0      the network decodes dt 0 only (the suite decodes -0.5 / 0 / +0.5 and scores all three; the dt 0 output is the same, the gate shows it)''',
    '''  LIMIT      number of frames, default 400 = all of the base set. Base set (--rule old, the default): the 400 frames of the old scoring
             (temp/soa_gap/map_ref_dev.py: the GaussianCaR logits cached by the lane project, val_tokens.json, 1706 frames in 43 of our 82 val scenes;
             every 4th of them) that are samples of this run's val set: the datasets of the current framework leave out the first and last keyframe
             of every scene, so 384 of the 400 remain (16 frames lost). --rule usable: every 4th of the CURRENT usable list (the old recipe, other
             frames); --rule old400 (with --no-model only): all 400 old frames, edge keyframes included. A smaller LIMIT spreads over the base set.
  --head     the FIRST LIMIT frames of the base set instead of spreading (a LIMIT run is then a prefix of the full run)
  --stored   only frames of the base set that are also in the run's stored 1000-frame suite evaluation (sanity gate: the per-frame counts of our model
             are compared with the stored ones); the check runs for every scored frame that is in that store anyway (--no-check: not)
  --no-model GaussianCaR and the prior only (no network; a couple of minutes for 400 frames): the check of the reference against the old numbers
  --all-dts  the network decodes -0.5 / 0 / +0.5 like the suite (needs the trainer's GT cache entry of every frame); default: dt 0 only, which is the
             same prediction at dt 0 (4 frames, both ways: identical counts) and a third faster''')
rep('''GT identity: the dataset's own signature must equal runs/RUN/gt.json (v8: 0.5_4_6a7aa7864a); a mismatch stops the script. Cache misses (GT, radar)
stop it too: the dataset would build the entry and WRITE it into data/cache/, and this script writes only OUT*.''',
    '''GT identity: the dataset's own signature must equal runs/RUN/gt.json (v8: 0.5_4_6a7aa7864a); a mismatch stops the script. The GT of a frame is the
trainer's cache file (what the suite scored against) or, where the cache has none (only 1463 of the 3109 val frames are cached for this signature), the
stored GT data/autogt/c643a7539d/<token>.npz of the same version (v9 code = v8 GT byte for byte, autogt/versions.csv same_gt_as; the script compares
12 frames that have both: every layer equal). A frame with neither, or without its radar cache entry, stops the script: the dataset would build and
WRITE it into data/cache/, and this script writes only OUT*.''')
open(p, "w", encoding="utf-8").write(s)
print("ok")
