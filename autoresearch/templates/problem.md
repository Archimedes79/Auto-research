# PROBLEM - <name>

Owned by the person (list it under "protected" in `problem.json`). Every agent reads this first. Keep it under two pages.

## Goal and constraints
- what the product must do, for whom, and what "better" means (quality AND cost, e.g. ms per frame)
- hard constraints (differentiable path, data licence, compute, TEST use)

## The product chain
input -> stage -> stage -> output (one line per stage, with its code folder)

## Metrics and protocol
- headline metrics and how they are read (which is the target, which are checks); the score, guards, splits, holdout and noise band are in
  `problem.json` and are not repeated here; seeds; the TEST rule: `TEST_LOG.md`

## Current best
The champion and its numbers: `champions/<name>/`. Here only what it is and what it lacks; no copied numbers.

## Known gaps (ranked)
1. ...

## Problem groups (the taxonomy every note, card and summary is filed under)
| id | group | question |
|---|---|---|

## Foundation topics (researched independently of the problem)
| id | topic | why it may matter here |
|---|---|---|

## Problem-specific workflow
- anything that deviates from `autoresearch/METHOD.md` (extra gates, special protocols)
