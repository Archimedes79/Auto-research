#!/bin/bash
for i in $(seq 1 115); do pgrep -f "diag_[u]nion" >/dev/null || break; sleep 5; done
pgrep -f "diag_[u]nion" >/dev/null && echo still running || echo done
