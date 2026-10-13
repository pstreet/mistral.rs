#!/usr/bin/env bash
# Liveness watchdog for mistralrs-server (user service).
#
# Restarts the server when it is wedged but not crashed (which
# Restart=on-failure never catches): the known signature is a poisoned
# ROCm context after an HSA memory fault (hip error 700), where every
# request 500s forever. Four signals are checked:
#   1. GET /health fails (process down or socket hung). Note /health is a
#      static handler, so a 200 here does NOT prove the GPU works.
#   2. New GPU-wedge lines in this service's journal since the last check.
#      A poisoned context only emits them when requests arrive, so signal
#      4 exists to manufacture traffic; without it an idle wedge resets
#      the fail counter every check (found live 2026-10-13: detected 1/2,
#      then silent for 40 min because no request came to fault).
#   3. The service being inactive/failed at all. A SIGTERM kill looks
#      identical to a deliberate stop (Result=success, status 15), so
#      systemd restarts nothing and the old check exited silently here.
#   4. A 1-token real generation against the default model every
#      PROBE_INTERVAL seconds while it is loaded (skipped when idle-
#      evicted, so the RAM-relief TTL still works). Probe failure feeds
#      the same fail counter, and its 500/hip-700 journal lines land in
#      the next check's window, so the two arms interlock: a poisoned
#      context is detected within ~2 checks even with zero real traffic.
# A wedge restart needs 2 consecutive failing checks (~2 min) and never
# fires within 4 min of a (re)start, so normal model loading can't trip
# it. An unexpected stop restarts immediately; use mistralrs-server-stop.sh
# for intentional stops so the watchdog stays quiet (marker file).
# Every action alerts via local mail plus ntfy (deduplicated per cause,
# 30 min cooldown). Knobs: MISTRALRS_ALERT_MAIL, MISTRALRS_NTFY_TOPIC
# (subscribe the topic in the ntfy app/web to receive pushes).
set -uo pipefail

SVC="mistralrs-server"
BASE_URL="http://127.0.0.1:1235" # must match [server] port in mistralrs.toml
URL="$BASE_URL/health"
CACHE_DIR="${XDG_CACHE_HOME:-$HOME/.cache}"
CURSOR_FILE="$CACHE_DIR/$SVC-health.cursor"
FAIL_FILE="$CACHE_DIR/$SVC-health.fails"
MARKER_FILE="$CACHE_DIR/$SVC-health.userstop"
ALERT_FILE="$CACHE_DIR/$SVC-health.alerted"
MAX_FAILS=2
MIN_UPTIME=240
ALERT_COOLDOWN=1800
MARKER_TTL=14400
ALERT_MAIL="${MISTRALRS_ALERT_MAIL:-$USER}"
NTFY_TOPIC="${MISTRALRS_NTFY_TOPIC:-mistralrs-strix-halo}"
PROBE_MODEL="${MISTRALRS_PROBE_MODEL:-qwen3.6-35b-q5}"
PROBE_INTERVAL=300
PROBE_TS_FILE="$CACHE_DIR/$SVC-health.probe"

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
    active)
        # The stop marker only means something while the service is down;
        # once it runs again, consume it so later unexpected stops alert.
        [ -f "$MARKER_FILE" ] && rm -f "$MARKER_FILE"
        ;; # wedge logic below
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
            now=$(date +%s)
            marker_age=$(( now - $(stat -c %Y "$MARKER_FILE" 2>/dev/null || echo 0) ))
            if [ "$marker_age" -lt "$MARKER_TTL" ]; then
                exit 0 # intentional stop, marker still fresh, stay quiet
            fi
            rm -f "$MARKER_FILE" # stale beyond the TTL: treat as unexpected
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

# Signal 4: real-generation probe, after the cursor save so its failure
# evidence lands in the next check's journal window (the interlock: a probe
# 500 from a poisoned context emits hip-700 lines the next journal sweep
# finds, so MAX_FAILS is reached even with zero real traffic).
if [ -z "$bad" ]; then
    last_probe=0
    [ -f "$PROBE_TS_FILE" ] && last_probe=$(cat "$PROBE_TS_FILE" 2>/dev/null || echo 0)
    now=$(date +%s)
    if [ $(( now - last_probe )) -ge "$PROBE_INTERVAL" ]; then
        echo "$now" > "$PROBE_TS_FILE"
        if curl -sf -m 10 "$BASE_URL/v1/models" 2>/dev/null \
            | jq -e --arg m "$PROBE_MODEL" \
                '.data[] | select(.id == $m and .status == "loaded")' >/dev/null 2>&1; then
            if ! curl -sf -m 60 "$BASE_URL/v1/completions" \
                -H 'Content-Type: application/json' \
                -d "{\"model\":\"$PROBE_MODEL\",\"prompt\":\"ok\",\"max_tokens\":1}" \
                >/dev/null 2>&1; then
                bad="model probe failed"
            fi
        fi
    fi
fi

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
