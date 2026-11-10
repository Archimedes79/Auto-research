cd "$(git rev-parse --show-toplevel)"
python3 autoresearch/tools/check_git.py --paths autoresearch/evalsuite bev_fusion/evaluation bev_fusion/tests/test_evaluation_api.py 2>&1 | tail -15
