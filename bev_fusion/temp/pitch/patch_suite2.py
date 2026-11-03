p = "evaluation/bev_suite.py"
s = open(p, encoding="utf-8").read()
s = s.replace('''def _task_only(run: str):
    """the Task of a run without loading its checkpoint (metrics definitions only)"""''', '''def _task_only(run: str, curves: bool = False, fusion: bool = False):
    """the Task of a run without loading its checkpoint (metrics definitions only; curves / fusion: the metric families of a store made with --curves / --fusion)"""''')
s = s.replace('''    return BEVTask(model.grid, model.spec, [g for g, _ in ns.NUSC_GROUPS], refine=refine)''', '''    task = BEVTask(model.grid, model.spec, [g for g, _ in ns.NUSC_GROUPS], refine=refine)
    task.curves, task.fusion = curves, fusion
    return task''')
s = s.replace('''    p.add_argument("--force", action="store_true", help="compare even when the GT signatures or (legacy records) the frame sets differ")
    c = p.parse_args(argv)
    from evalsuite.quantitative import Quantitative
    name = f"eval{c.tag}_n{c.n}"''', '''    p.add_argument("--force", action="store_true", help="compare even when the GT signatures or (legacy records) the frame sets differ")
    p.add_argument("--curves", action="store_true", help="compare the stores made with `eval --curves` (PR / ROC / calibration / velocity numbers are bootstrapped too)")
    p.add_argument("--fusion", action="store_true", help="compare the stores made with `eval --fusion` (detf / detfp / cov)")
    c = p.parse_args(argv)
    from evalsuite.quantitative import Quantitative
    name = f"eval{c.tag}{'_curves' if c.curves else ''}{'_fusion' if c.fusion else ''}_n{c.n}"''')
s = s.replace("    task = _task_only(c.run_a)\n", "    task = _task_only(c.run_a, c.curves, c.fusion)\n")
open(p, "w", encoding="utf-8").write(s)
