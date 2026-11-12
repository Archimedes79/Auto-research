#!/bin/bash
ps aux | grep -E "[t]est_lane_prototype|[t]est_eval_parts" | cut -c1-150
