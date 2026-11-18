#!/bin/bash
cd "$(dirname "$0")/../.."
bash temp/e379/read379.sh dump d_s1 >> temp/e379/read_d_s1.out 2>&1
bash temp/e379/read379.sh cpu d_s1 >> temp/e379/read_d_s1.out 2>&1
