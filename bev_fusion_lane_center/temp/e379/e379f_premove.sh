# scratch: before the move: the scratch copies parse, the live files are the v1 copies, no trainer job runs
/home/ml/miniconda3/envs/gcar/bin/python -c "import ast, sys; [ast.parse(open(f).read()) for f in sys.argv[1:]]; print('parse ok')" temp/e379/e379f_jtoken.py temp/e379/e379f_jtoken_train.py temp/e379/e379f_test_jtoken.py
sha256sum -c temp/e379/e379f_v2.sha256
echo "trainer jobs running: $(ps aux | grep -E '[j]token_train|[j]live_train|[j]past_train|[j]map_train' | wc -l)"
date +%H:%M:%S
