S = "bev_fusion_lane_center/temp/review_session_scratch_2026-11-01/"
def sub(path, old, new):
    s = open(path, encoding="utf-8").read()
    assert s.count(old) == 1, (path, old[:60], s.count(old))
    open(path, "w", encoding="utf-8", newline="\n").write(s.replace(old, new))
    print("ok", path)
old = "`plan.py` (status of the plan nodes from the schedule and the ledger), "
new = old + "`commit_paths.py` (commit exactly the named paths in the shared checkout: temporary index, compare-and-swap on HEAD, scratch and forbidden files refused), "
sub(S + "ar_readme_head8.md", old, new)
sub("autoresearch/README.md", old, new)
sub("CLAUDE.md", "own paths only (a temporary index when other chats work in the checkout)", "own paths only (`autoresearch/tools/commit_paths.py`: a temporary index, so what other chats stage or edit stays out)")
