# Loop prompt (P3 + P4) - paste after `/loop` and fill the <...>

## Start a loop (do this once, before pasting the prompt)
All commands run from the problem folder (e.g. `bev_fusion_lane_center/`) through the WSL: `wsl python3 ../autoresearch/tools/<tool>.py ...` (the tools find the problem from `autoresearch/config.json` or `AR_PROJECT`; `tools/...` in this folder's other files means `autoresearch/tools/`, not the problem's own `tools/`).
1. **P2 is done** (`research_steps.md` steps 3-4): `PLAN.md` from `templates/plan.md`; one folder `experiments/E<id>_<slug>/` per card with `experiment.md` and, for runs the tools decide, `experiment.json` (`experiment.py check E<id>` says ready); schedule rows (`schedule.py add <item> ... --command "..."`); a planned ledger row (`ledger.py add <run id> --hyp --pred --rule`) for every run, also the arms of an `experiment.json` (the launcher reads the card from the ledger row or from `--hyp / --pred / --rule`, not from the experiment file). `run.py` refuses a run without a card.
2. **The last loop is filed**: its P4 list below is done, `clean_temp.py loop<NN-1>` (dry run) has nothing left, `check_structure.py` reports 0 problems, the problem's tests pass.
3. **Number and range**: loop `<NN>` = last loop + 1; its experiment ids continue after the highest `experiments/E<id>_*`. The range belongs in `project.json` "loops" (`metrics.py`, `ledger.py tail --loop`); `project.json` is protected, so ask through the decisions file (until then the loop's runs count under the last open range).
4. **Tools ready**: `install_agents.py --check` (the `ar-*` agents), the gate hook in the workspace's `.claude/settings.json`, git identity set (`run.py` refuses otherwise).
5. **Fill the prompt below** and save the filled copy with the problem's records (lane problem: `experiment_execution/history/prompts/PROGRAM_loop<NN>.md`); rewrite `HANDOVER.md`: loop `<NN>`, this chat is the lead and owns the code, scratch in `temp/loop<NN>/`.
6. **Start** in a Claude Code session opened in the workspace folder (the parent of `autoresearch/`, so the agents and the hook load): `/loop` without an interval (the loop paces itself; wake-ups below), followed by the filled prompt.

## The prompt

STOP_AT: <day time> (a work window: keep going until then) | SHUTDOWN: <yes / no; only if no other chat runs> | MAIL: <no / address and how> | TEST: only inside a champion decision recorded in the decisions file

Problem folder: <path>. Loop <NN>. Plan, schedule, ledger, decisions: as named in `HANDOVER.md`. Method: `autoresearch/METHOD.md` P3 and P4. This chat is the lead and owns the code for the loop (say so in `HANDOVER.md`). Do not re-plan; a change of direction goes through `ar-planner` as a new schedule version. Tools: `wsl python3 ../autoresearch/tools/<tool>.py` from the problem folder.

Every wake-up (when a run record changes, else every 20-25 min)
- a training whose trainer follows the `trainsuite` contract (bev_fusion): write the card's kill criterion into its run folder's `kill.txt` and keep `python -m trainsuite --root <runs> watch <run> --enforce` running as a Monitor; other trainers: check the kill criterion by hand at each wake-up
- `schedule.py show` and `ledger.py tail --loop <NN>` (state is read from the run records; run them in the WSL, the namespace of the launchers; when `tail` reports runs without a ledger row, `ledger.py sync` adds them), then start the ready items through `run.py` (a chain with `--after`; `schedule.py next` lists what is ready); read the open decisions
- do what needs a chat: detail plan (`ar-planner` mode detail), code (`ar-coder`; `ar-critic` mode code for shared code), decide (`decide.py --experiment E<id> --write` when the cards of an experiment are complete), summary (section 3 of the card's `experiment.md`); a run needs its card (a planned ledger row or `--hyp / --pred / --rule`), run.py refuses otherwise
- at most <n> GPU jobs; checkpoint and resume; never kill another chat's jobs
- write schedule, ledger notes, plan status, HANDOVER (<= 3 KB, rewrite); next wake-up

At STOP_AT (P4)
- end-of-loop section of the plan; report from the experiment summaries; `check_numbers.py`; `ar-critic` mode numbers; `ar-planner` mode tracking (the best versions, a promotion proposal, the next round's open branches)
- card statuses, closed questions, open points for the next cycle; decisions for the person into the decisions file
- commit the experiment code (scratch included), then `clean_temp.py loop<NN> --move`; `check_structure.py`; tests
- HANDOVER rewritten; mail / shutdown only as allowed above
