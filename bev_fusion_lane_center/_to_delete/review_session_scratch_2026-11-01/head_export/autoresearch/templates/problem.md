# PROBLEM - <name>

Owned by the person. Every agent reads this first. Keep it under two pages; details go into the linked files.

## Goal and constraints
- what the product must do, for whom, and what "better" means (quality AND cost, e.g. ms per frame)
- hard constraints (differentiable path, data licence, compute: one GPU, budget per loop, TEST usage)

## The product chain
input -> stage -> stage -> output (one line per stage, with the code folder and the current best block)

## Metrics and protocol
- headline metrics and how they are read (which is the target, which are checks); the evaluation entry point; splits; seeds; noise band; TEST rule -> link to the protocol file

## Current best (with numbers, split, date)
| system | metric 1 | metric 2 | ms | card |
|---|---|---|---|---|

## Known gaps (ranked)
1. ...

## Problem groups (the taxonomy every note, card and summary is filed under)
| id | group | question |
|---|---|---|

## Foundation topics (researched independently of the problem)
| id | topic | why it may matter here |
|---|---|---|

## Problem-specific workflow
- folder mapping (research, training, evaluation, experiment summaries, execution state)
- anything that deviates from `autoresearch/METHOD.md` (extra gates, special protocols)
