From the problem folder (`problems/toy`; until 2026-11-26 `autoresearch/examples/toy`), under WSL:

```
bash tools/prepare.sh make training/configs/base.json       # raw shards and ground truth
bash tools/py.sh gt/generate.py view --out runs/gt/truth.png # the ground-truth picture of section 3
python3 experiments/E1_width/launch.py                       # the four runs, through Training and run.py
(cd ../../autoresearch && AR_PROJECT=../problems/toy python3 tools/decide.py --experiment E1)     # the verdict
bash tools/py.sh reports/E1/cost.py                          # the cost numbers of this report
bash tools/py.sh ../../reporter/reporter.py build reports/E1 --strict
```

A repeat needs new run ids (a finished id is refused). The code of each run is the commit of its tag `toy/E1.<arm>_s<seed>`; the runs execute the working tree of that moment, not the tag.
