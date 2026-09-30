#!/usr/bin/env bash
# Intentional stop for mistralrs-server: leaves a marker so the liveness
# watchdog knows the stop was deliberate and stays quiet instead of
# alerting + restarting. Always use this instead of systemctl stop.
set -uo pipefail

SVC="mistralrs-server"
CACHE_DIR="${XDG_CACHE_HOME:-$HOME/.cache}"

mkdir -p "$CACHE_DIR"
touch "$CACHE_DIR/$SVC-health.userstop"
systemctl --user stop "$SVC"
echo "$SVC stopped (watchdog notified to stay quiet)"
