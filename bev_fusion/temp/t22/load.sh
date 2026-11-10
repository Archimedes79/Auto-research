#!/bin/bash
nproc; cat /proc/loadavg; free -g | sed -n 2p
