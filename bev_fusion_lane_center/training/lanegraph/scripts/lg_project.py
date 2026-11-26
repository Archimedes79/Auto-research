"""Access to the project's own code for the scripts and tests: the scorer and the ORIGINAL modules of the system of record. The lanegraph modules
themselves never import it.

    use_project_code()                         # every project code folder on sys.path (the project imports flat by name, as tools/env/bflc_env.sh)
    qgraph, jends = original_models(seed)      # QGraph of B (runs/e273_p_s<seed>) and JEndModel of the junction run (runs/e324_s_s<seed>), CPU
    raw, kinds = original_graph(pred, e, attached)   # the card's read-out of one frame (qends_eval.cmd_score: --snap, or --attach --snap --snap-nogeom)
"""
import functools
import os
import sys

import numpy as np

from lg_config import PROJECT


@functools.cache
def use_project_code() -> None:
    dirs = [PROJECT.parent] + [d for top in ("net", "training", "evaluation", "tools") for d in [PROJECT / top, *(PROJECT / top).rglob("*")]
                               if d.is_dir() and "__pycache__" not in d.parts and not any(p.startswith("_session_scratch_") for p in d.parts)]
    for d in dirs:
        if str(d) not in sys.path:
            sys.path.append(str(d))


def original_models(seed: int, device: str = "cpu"):
    use_project_code()
    import jends_eval as JEV
    import qpredict as QP
    return QP.load_model(os.path.join(PROJECT, "runs", f"e273_p_s{seed}"), device), JEV.load_model(f"e324_s_s{seed}", device)[0]


def original_graph(pred: dict, e: dict, attached: bool) -> tuple:
    """-> (the raw graph dict of the scorer, the kinds of its pieces in our names: lane / link / connector)"""
    use_project_code()
    import jwire as JW
    import qends_eval as QEV
    import qgraph_eval as QE
    import qgraph_readout as QR
    pr = QR.apply(pred, QR.parse(QEV.DEFAULT_SPEC))
    graph, bk = QE.readout(pr, 0.5, 0.5)
    p = 1.0 / (1.0 + np.exp(-e["logit"])) * e["allowed"]
    ia, ib = np.nonzero(p >= 0.1)
    pairs = QEV.make_pairs(e, ia, ib, e["kp_pos"].shape[0] // 2, attach=attached)
    long = np.asarray(pr["links"]["long"], bool)
    raw = QEV.add_connectors(graph.lines, graph.edges, graph.nodes, bk["kind"], bk["piece_links"], long, pairs, snap=True, snap_nogeom=attached)[0]
    keep = JW.kept_ends(graph.lines, bk["kind"], bk["piece_links"], long, pairs, JW.cover_pos())[0]
    kinds = [{"section": "lane", "link": "link"}[bk["kind"][i]] for i in keep] + ["connector"] * len(pairs)
    return raw, kinds
