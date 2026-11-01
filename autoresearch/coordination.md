# Coordination between chats (design, 2026-11-01)

Mirko, in chat: improve the coordination between chats; the hold marker "makes sense". Status: the hold marker is built (`tools/hold.py`, 2026-11-01: `run.py` refuses a held id and sets a hold at exit 75, `ledger.py` / `schedule.py` know the state `held`); claims are design, not built.

## What went wrong on 2026-11-01
1. A stopped run (`t10.dual_long_s0`, exit 75) was started again by another chat on an instruction that was passed on second-hand, after a later word of the person had said stop. Nothing in the files said "this run is on hold".
2. Freezes ("do not edit `training/*.py` and the autogt files until about 17:10") lived only in chat messages and went stale (17:10 became "no date", then "lifted").
3. Messages between chats are queued and not confirmed read.
4. Who owns which files was in the chats' heads; two chats nearly wrote the same tool (`provenance.py`).

## Two small mechanisms (files and one tool each, no daemon)
**1. Hold marker.** A file `<run_records>/<run id>/HOLD` (one JSON object: who, when, why, the person's words).
- Written when a run ends with exit 75 (stopped), or by `hold.py set <id> --by <chat> --reason "..." --words "<the person's words>"`.
- `run.py` refuses to start a run id that has a HOLD (status `held`, the reason is printed); chains with `--after` stop at it.
- `hold.py clear <id> --words "<the person's words>"` removes it and adds a ledger note with the words. Only the person's words clear it, never a chat's own judgement.
- `hold.py show` lists the holds.
This is the enforcement of hard rule 6 (the person's decisions are acted on only when recorded) for runs.

**2. Claims.** A file `autoresearch/state/claims.json` (not in git, like the machine lock): entries {chat, paths, kind (own or freeze), reason, since, until or "until released by <chat>"}.
- `claims.py add | release | show | check <path>`; `check_git.py` and the commit helpers warn when a path is claimed by another chat.
- A claim is advice: it blocks nothing, but it is written down, has a reason and a release, and shows who to ask. The freeze of 2026-11-01 would have been one claim.
- Built at the second use (a second freeze that goes stale), not before.

## Rules that stay as they are
A relayed instruction of the person is acted on only with their words quoted and a record (DECISIONS or the handover). Messages between chats stay as they are: no message bus, no lock server.

## Build
Tools owner: `hold.py` (about 80 lines), a hunk in `run.py` for the refusal and for writing the HOLD at exit 75, a selftest that fails without the refusal, a critic pass. The ledger row of `t10.dual_long_s0` (failed, exit 75) becomes "stopped, on hold".
