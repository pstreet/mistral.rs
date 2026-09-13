#!/usr/bin/env bash
# Liveness watchdog for mistralrs-server (user service).
#
# Restarts the server when it is wedged but not crashed (which
# Restart=on-failure never catches): the known signature is a poisoned
# ROCm context after an HSA memory fault (hip error 700), where every
# request 500s forever. Two signals are checked:
#   1. GET /health fails (process down or socket hung). Note /health is a
#      static handler, so a 200 here does NOT prove the GPU works.
#   2. New GPU-wedge lines in this service's journal since the last check.
# A restart needs 2 consecutive failing checks (~2 min) and never fires
# within 4 min of a (re)start, so normal model loading can't trip it.
# It also never starts a service you stopped yourself (checks is-active).
set -uo pipefail

SVC="mistralrs-server"
URL="http://127.0.0.1:1235/health" # must match [server] port in mistralrs.toml
CACHE_DIR="${XDG_CACHE_HOME:-$HOME/.cache}"
CURSOR_FILE="$CACHE_DIR/$SVC-health.cursor"
FAIL_FILE="$CACHE_DIR/$SVC-health.fails"
MAX_FAILS=2
MIN_UPTIME=240

mkdir -p "$CACHE_DIR"
exec 9>"$CACHE_DIR/$SVC-health.lock"
flock -n 9 || exit 0 # another check still running

[ "$(systemctl --user is-active "$SVC")" = "active" ] || exit 0

started=$(systemctl --user show "$SVC" -p ActiveEnterTimestamp --value)
if [ -n "$started" ]; then
    started_s=$(date -d "$started" +%s 2>/dev/null || echo 0)
    [ $(( $(date +%s) - started_s )) -ge "$MIN_UPTIME" ] || exit 0
fi

fails=0
[ -f "$FAIL_FILE" ] && fails=$(cat "$FAIL_FILE" 2>/dev/null || echo 0)

bad=""
curl -sf -m 10 "$URL" >/dev/null 2>&1 || bad="health-check failed"
if [ -f "$CURSOR_FILE" ]; then
    scope="--after-cursor=$(cat "$CURSOR_FILE" 2>/dev/null)"
else
    scope="--since=3 minutes ago"
fi
if journalctl --user -u "$SVC" --no-pager $scope 2>/dev/null \
    | grep -qE "hip error 700|HSA_STATUS_ERROR_MEMORY_FAULT|Failed to reset model cache"; then
    bad="${bad:+$bad; }GPU-wedge signature in journal"
fi
cursor=$(journalctl --user -u "$SVC" --no-pager --show-cursor -n0 2>/dev/null \
    | tail -n1 | sed 's/^-- cursor: //')
[ -n "$cursor" ] && echo "$cursor" > "$CURSOR_FILE"

if [ -n "$bad" ]; then
    fails=$((fails + 1))
    echo "$fails" > "$FAIL_FILE"
    echo "$SVC unhealthy ($fails/$MAX_FAILS): $bad"
    if [ "$fails" -ge "$MAX_FAILS" ]; then
        echo 0 > "$FAIL_FILE"
        echo "$SVC unhealthy $fails consecutive checks, restarting"
        systemctl --user restart "$SVC"
    fi
else
    echo 0 > "$FAIL_FILE"
fi
