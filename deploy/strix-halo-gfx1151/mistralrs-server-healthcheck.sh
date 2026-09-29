#!/usr/bin/env bash
# Liveness watchdog for mistralrs-server (user service).
#
# Restarts the server when it is wedged but not crashed (which
# Restart=on-failure never catches): the known signature is a poisoned
# ROCm context after an HSA memory fault (hip error 700), where every
# request 500s forever. Three signals are checked:
#   1. GET /health fails (process down or socket hung). Note /health is a
#      static handler, so a 200 here does NOT prove the GPU works.
#   2. New GPU-wedge lines in this service's journal since the last check.
#   3. The service being inactive/failed at all. A SIGTERM kill looks
#      identical to a deliberate stop (Result=success, status 15), so
#      systemd restarts nothing and the old check exited silently here.
# A wedge restart needs 2 consecutive failing checks (~2 min) and never
# fires within 4 min of a (re)start, so normal model loading can't trip
# it. An unexpected stop restarts immediately; use mistralrs-server-stop.sh
# for intentional stops so the watchdog stays quiet (marker file).
# Every action alerts via local mail plus ntfy (deduplicated per cause,
# 30 min cooldown). Knobs: MISTRALRS_ALERT_MAIL, MISTRALRS_NTFY_TOPIC
# (subscribe the topic in the ntfy app/web to receive pushes).
set -uo pipefail

SVC="mistralrs-server"
URL="http://127.0.0.1:1235/health" # must match [server] port in mistralrs.toml
CACHE_DIR="${XDG_CACHE_HOME:-$HOME/.cache}"
CURSOR_FILE="$CACHE_DIR/$SVC-health.cursor"
FAIL_FILE="$CACHE_DIR/$SVC-health.fails"
MARKER_FILE="$CACHE_DIR/$SVC-health.userstop"
ALERT_FILE="$CACHE_DIR/$SVC-health.alerted"
MAX_FAILS=2
MIN_UPTIME=240
ALERT_COOLDOWN=1800
ALERT_MAIL="${MISTRALRS_ALERT_MAIL:-$USER}"
NTFY_TOPIC="${MISTRALRS_NTFY_TOPIC:-mistralrs-strix-halo}"

mkdir -p "$CACHE_DIR"
exec 9>"$CACHE_DIR/$SVC-health.lock"
flock -n 9 || exit 0 # another check still running

alert() { # alert <dedup-key> <subject> <message>
    local key="$1" subject="$2" msg="$3" now last
    now=$(date +%s)
    if [ -f "$ALERT_FILE" ]; then
        read -r last_key last_ts < "$ALERT_FILE" 2>/dev/null || true
        if [ "${last_key:-}" = "$key" ] && [ $(( now - ${last_ts:-0} )) -lt "$ALERT_COOLDOWN" ]; then
            return 0
        fi
    fi
    echo "$key $now" > "$ALERT_FILE"
    echo "$msg" | mail -s "[mistralrs] $subject" "$ALERT_MAIL" 2>/dev/null || true
    curl -sf -m 10 -H "Title: [mistralrs] $subject" -d "$msg" \
        "https://ntfy.sh/$NTFY_TOPIC" >/dev/null 2>&1 || true
    echo "alert sent ($key): $subject"
}

state=$(systemctl --user is-active "$SVC" 2>/dev/null || echo unknown)
case "$state" in
    active) ;; # wedge logic below
    activating|deactivating|reloading) exit 0 ;; # transient, check next minute
    failed)
        rm -f "$MARKER_FILE" # a crash is never an intentional stop
        echo 0 > "$FAIL_FILE"
        if systemctl --user restart "$SVC" 2>/dev/null; then
            echo "$SVC was in failed state, restarted"
            alert failed "$SVC crashed, restarted" "$SVC entered the failed state and was restarted by the watchdog."
        else
            alert restart-failed "$SVC crashed, RESTART FAILED" "$SVC is in the failed state and 'systemctl --user restart' failed. Manual intervention needed."
        fi
        exit 0 ;;
    *)
        if [ -f "$MARKER_FILE" ]; then
            rm -f "$MARKER_FILE"
            exit 0 # intentional stop, stay quiet
        fi
        echo 0 > "$FAIL_FILE"
        if systemctl --user start "$SVC" 2>/dev/null; then
            echo "$SVC was down (state: $state), started"
            alert down "$SVC was down, started" "$SVC was found in state '$state' (not an intentional stop) and was started by the watchdog."
        else
            alert start-failed "$SVC is DOWN, START FAILED" "$SVC was found in state '$state' and 'systemctl --user start' failed. Manual intervention needed."
        fi
        exit 0 ;;
esac

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
        if systemctl --user restart "$SVC" 2>/dev/null; then
            alert wedge "$SVC wedged, restarted" "$SVC failed $fails consecutive checks ($bad) and was restarted by the watchdog."
        else
            alert restart-failed "$SVC wedged, RESTART FAILED" "$SVC failed $fails consecutive checks ($bad) and 'systemctl --user restart' failed. Manual intervention needed."
        fi
    fi
else
    echo 0 > "$FAIL_FILE"
fi
