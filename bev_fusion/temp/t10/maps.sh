#!/bin/bash
cd "$(dirname "$0")/../.." || exit 1
~/miniconda3/envs/gcar/bin/python temp/t10/maps.py
