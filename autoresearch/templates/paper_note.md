# <paper id> <short title>

<!-- One note per paper, with its code, in its own subfolder of the topic (topics/<slug>/<paper id>/note.md), written by one agent. The eight lines below are the header that tools/topic_table.py turns into the table of the topic: exactly these keys, in this order, one per line, directly after the title line, nothing between them; delete this comment. Links are plain (no spaces, quotes, brackets, parentheses, pipes: percent-encode them). -->

- title: <the title of the paper>
- relevance: <0-3; 3 = take it into the plan now>
- summary: <ONE line, about three sentences and at most 700 characters: what it does and why it matters for us>
- status: <checked (page or full text read, with the URL in the body) / digest (from a summary) / memory>
- paper: <https link to the paper or its project page>
- code: <https link to the repository, or ->
- verdict: <use / watch / context / not relevant>
- result: -

| | |
|---|---|
| venue / year | <venue, year> |
| groups | <group id: relevance 0-3>, ... |
| licence | <licence of the code, or none> |
| why the verdict | <one line> |

**What it does** (3-5 sentences, mechanism level): ...

**Results that matter** (numbers with dataset and metric): ...

**Transferable technique** (what we could take; the K id if registered): ...

**Why it performs** (the ingredients that carry the gain: the paper's ablation, or say that none is shown): ...

**Cost and fit** (the price in compute, memory, latency, data, complexity; is the gain worth it for us; what it needs that we do not have): ...

**Doubts** (what is not shown, weak baselines, leakage risks): ...
