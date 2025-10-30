#!/usr/bin/env bash
# Control script to run the Raspberry device component
# Usage: ./device_control.sh {start|start_background|stop|restart|status|logs|fg}

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR%/scripts}"
cd "$REPO_ROOT"

VENV_PY="$REPO_ROOT/.venv/bin/python"
DEVICE_MODULE="atlantico_rpi.device"

PID_DIR="$REPO_ROOT/run/pids"
LOG_DIR="$REPO_ROOT/run/logs"
PID_FILE="$PID_DIR/device.pid"
LOG_FILE="$LOG_DIR/device.log"

mkdir -p "$PID_DIR" "$LOG_DIR"

start_background() {
  if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "device is already running (pid $(cat "$PID_FILE"))"
    return 0
  fi

  echo "Starting device in background (module: $DEVICE_MODULE)"
  if [ -x "$VENV_PY" ]; then
    nohup "$VENV_PY" -m "$DEVICE_MODULE" "$@" >>"$LOG_FILE" 2>&1 &
  else
    nohup python3 -m "$DEVICE_MODULE" "$@" >>"$LOG_FILE" 2>&1 &
  fi
  echo $! > "$PID_FILE"
  sleep 0.2
  echo "Started device (pid $(cat "$PID_FILE")), logs: $LOG_FILE"
}

start() {
  if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "device appears to be running (pid $(cat "$PID_FILE")); stop it first or use start_background."
    return 1
  fi
  echo "Starting device in foreground"
  if [ -x "$VENV_PY" ]; then
    exec "$VENV_PY" -m "$DEVICE_MODULE" "$@"
  else
    exec python3 -m "$DEVICE_MODULE" "$@"
  fi
}

stop() {
  if [ -f "$PID_FILE" ]; then
    pid=$(cat "$PID_FILE")
    echo "Stopping device (pid $pid)"
    kill "$pid" || true
    rm -f "$PID_FILE"
  else
    echo "No PID file found; device may not be running"
  fi
}

status() {
  if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "device running (pid $(cat "$PID_FILE"))"
    tail -n 20 "$LOG_FILE" || true
    return 0
  else
    echo "device not running"
    return 1
  fi
}

logs() {
  if [ -f "$LOG_FILE" ]; then
    tail -n 200 -f "$LOG_FILE"
  else
    echo "No log file yet: $LOG_FILE"
    exit 1
  fi
}

case ${1:-} in
  start) shift; start "$@" ;; 
  start_background) shift; start_background "$@" ;; 
  stop) stop ;; 
  restart) stop; start_background ;; 
  status) status ;; 
  logs) logs ;; 
  fg)
    # run in foreground and stream logs to stdout
    echo "Running device in foreground (see logs in $LOG_FILE)"
    if [ -x "$VENV_PY" ]; then
      "$VENV_PY" -m "$DEVICE_MODULE" "$@" 2>&1 | tee "$LOG_FILE"
    else
      python3 -m "$DEVICE_MODULE" "$@" 2>&1 | tee "$LOG_FILE"
    fi
    ;;
  test_train)
    echo "Running training test script using: ${VENV_PY}"
    SCRIPT="$REPO_ROOT/scripts/test_train.py"
    if [ -x "${VENV_PY}" ]; then
      "${VENV_PY}" "$SCRIPT" "${@:2}"
    else
      python3 "$SCRIPT" "${@:2}"
    fi
    ;;
  *) echo "Usage: $0 {start|start_background|stop|restart|status|logs|fg|test_train} [-- <args passed to module>]"; exit 2 ;;
esac
