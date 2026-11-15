#!/bin/bash
ps -eo pid,cmd | grep -E "distill_student|window_gate|start_A2" | grep -v grep | cut -c1-220
date
