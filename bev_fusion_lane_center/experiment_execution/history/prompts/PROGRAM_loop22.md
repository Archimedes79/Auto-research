# PROGRAM loop 22 (filled 2026-10-31 23:40 from `../autoresearch/prompts/loop.md`; decision D28)

STOP_AT: 2026-11-01 06:30 (a work window; jobs in flight may run 2 - 3 h longer, Mirko: "Be flexible, 2-3 h more is ok") | SHUTDOWN: ~~yes, after the summaries~~ CANCELLED (2026-11-01 ~03:00, Mirko via the BEV system chat: "Yes: cancel the 09:30 shutdown; the machine stays on until the 6000-step run and its evaluation are done"; D28 amended); at the end of the round tell the Architecture chat that the quiet window for its merge has begun | MAIL: no (summary in files only) | TEST: one check, the final lane prototype against the control, logged in `evaluation/TEST_USAGE_LOG.md` first (Mirko: "One TEST check"); no promotion without Mirko (D4 stays open)

Problem folders: `bev_fusion_lane_center/` (loop 22, E261 ff.) and `bev_fusion/` (overnight round, cards `t11_*`). Lead and code owner for both tonight: the repository-review chat (session 6e9638e0); the BEV chats are idle. Plan, schedule, ledger, decisions: as named in each `HANDOVER.md` / `records/README.md`. Method: `autoresearch/METHOD.md` P2 (light cycle: hypotheses with three lenses, referee, plan) then P3 and P4. Tools: `wsl python3 ../autoresearch/tools/<tool>.py` from the problem folder (`AR_PROJECT=../bev_fusion` for bev_fusion).

Goal (D28): a ready-to-use prototype with good performance for each problem by the morning.
- lane: package the current best fast system (`recall_dp`) as one command (`tools/prototype/lane_prototype.py`: planner-ready graph JSON, numbers, video), then attack gap 1 (coverage at range: lanes found 0.767 / R@1 far 0.771 vs the control's 0.800 / 0.811) at Q >= 0.735, <= 14.3 vertices per 100 m, fast-chain speed; adopt only by `decide.py`.
- bev_fusion: the best open arm of the board (`t10_dual`), its seed twin first (its rule E1), then its weakest head (the motion term does not move: val group.motion 1.094 at every step of every run); packaged as one command; `records/champion.json` "promoted" stays null.

GPU: one bev_fusion training fills the 12 GB card (11.2 GB), so the two slots run one training at a time: bev_fusion first (Mirko: "bev_fusion first"), the lane GPU arms after it; CPU arms on cached raw outputs run in parallel (2 threads, check the load).

Every wake-up (a run record changes, else every 20 - 25 min): `schedule.py show`, `ledger.py tail --loop 22` (bev: `AR_PROJECT=../bev_fusion ... ledger.py tail`), start what is ready, decide finished experiments (`decide.py --experiment E<id> --write`; bev: the card's rule with `board.py` and `bev_suite.py compare`), write ledger notes, plan status, HANDOVER.

At STOP_AT (P4): end-of-loop section of the plan, morning summary page per problem (results, how to run the prototype, images), `check_numbers.py`, an `ar-reviewer` pass in mode numbers on the summaries, `check_structure.py`, HANDOVER; then shutdown when nothing of ours runs.
