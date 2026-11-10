cd "$(git rev-parse --show-toplevel)"
python3 autoresearch/tools/check_git.py --paths $(cat bev_fusion/temp/pitch3/paths.txt) 2>&1 | tail -3
python3 autoresearch/tools/commit_paths.py -F bev_fusion/temp/pitch3/commit_msg.txt $(cat bev_fusion/temp/pitch3/paths.txt) 2>&1 | tail -4
