#!/bin/bash
# names of the environment variables of the t10.dual_long_s0 launcher and the values of the non-secret run variables (who started it?)
pid=$(pgrep -f "tools/run.py --id t10.dual_long_s0 " | head -1)
echo "launcher pid: $pid"
tr '\0' '\n' < /proc/$pid/environ 2>/dev/null | cut -d= -f1 | grep -E '^(AR_|CLAUDE|WSL|WT_|TERM|SSH|VSCODE|PWD|USER|HOME|OLDPWD|SHLVL|_)' | tr '\n' ' '
echo
tr '\0' '\n' < /proc/$pid/environ 2>/dev/null | grep -E '^(AR_LOCK_OWNER|AR_PROJECT|AR_HEARTBEAT_S|PWD|OLDPWD|WSL_DISTRO_NAME|CLAUDE_CODE_ENTRYPOINT|CLAUDE_SESSION_ID|AR_SESSION)=' | cut -c1-160
echo "---- machine lock:"
cd "$(dirname "$0")/../../../autoresearch/tools" && python3 machine_lock.py show 2>&1 | head -8
