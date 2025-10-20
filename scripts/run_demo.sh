#!/usr/bin/env bash
set -euo pipefail
# Simple helper to run the legacy server and the Raspberry device locally
# Usage: ./run_demo.sh start|stop|status|logs

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
SERVER_SCRIPT="$ROOT_DIR/../atlantico-server/server.py"
DEVICE_MODULE="atlantico_rpi.device"
VENV_PY="$ROOT_DIR/.venv/bin/python"

LOG_DIR="$ROOT_DIR/run/logs"
PID_DIR="$ROOT_DIR/run/pids"
mkdir -p "$LOG_DIR" "$PID_DIR"

start() {
  if [ ! -f "$SERVER_SCRIPT" ]; then
    echo "Server script not found at $SERVER_SCRIPT"
    exit 1
  fi

  # Check broker
  if ! timeout 1 bash -c "</dev/tcp/127.0.0.1/1883" 2>/dev/null; then
    echo "Warning: MQTT broker not reachable at 127.0.0.1:1883"
  fi

  # Start server (prefer server venv python if available)
  SERVER_VENV_PY="$ROOT_DIR/../atlantico-server/.venv/bin/python"
  if [ -f "$PID_DIR/server.pid" ]; then
    echo "Server appears to be running (pid file exists)"
  else
    echo "Starting server -> $LOG_DIR/server.log"
    if [ -x "$SERVER_VENV_PY" ]; then
      nohup "$SERVER_VENV_PY" "$SERVER_SCRIPT" > "$LOG_DIR/server.log" 2>&1 &
    else
      nohup python3 "$SERVER_SCRIPT" > "$LOG_DIR/server.log" 2>&1 &
    fi
    echo $! > "$PID_DIR/server.pid"
    sleep 0.2
  fi

  # Start device using the venv python if available
  if [ -f "$PID_DIR/device.pid" ]; then
    echo "Device appears to be running (pid file exists)"
  else
    echo "Starting device -> $LOG_DIR/device.log"
    if [ -x "$VENV_PY" ]; then
      # run module via the venv python to ensure local package import works
      nohup "$VENV_PY" -m atlantico_rpi.device --connect --run-for 0 > "$LOG_DIR/device.log" 2>&1 &
    else
      nohup python3 -m atlantico_rpi.device --connect --run-for 0 > "$LOG_DIR/device.log" 2>&1 &
    fi
    echo $! > "$PID_DIR/device.pid"
    sleep 0.2
  fi

  echo "Started. Logs: $LOG_DIR"
}

stop() {
  for svc in server device; do
    pidfile="$PID_DIR/${svc}.pid"
    if [ -f "$pidfile" ]; then
      pid=$(cat "$pidfile" 2>/dev/null || true)
      if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
        echo "Stopping $svc (pid=$pid)"
        kill "$pid" || true
        sleep 0.2
      fi
      rm -f "$pidfile"
    else
      echo "$svc not running (no pidfile)"
    fi
  done
}

status() {
  for svc in server device; do
    pidfile="$PID_DIR/${svc}.pid"
    if [ -f "$pidfile" ]; then
      pid=$(cat "$pidfile" 2>/dev/null || true)
      if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
        echo "$svc running (pid=$pid)"
      else
        echo "$svc pidfile exists but process not running"
      fi
    else
      echo "$svc not running"
    fi
  done
}

logs() {
  tail -n 200 -f "$LOG_DIR/server.log" "$LOG_DIR/device.log"
}

case "${1:-}" in
  start) start ;; 
  stop) stop ;; 
  status) status ;; 
  logs) logs ;; 
  fg-server)
    # run server in foreground, tee to server.log
  SERVER_VENV_PY="$ROOT_DIR/../atlantico-server/.venv/bin/python"
    if [ -x "$SERVER_VENV_PY" ]; then
      "$SERVER_VENV_PY" "$SERVER_SCRIPT" 2>&1 | tee "$LOG_DIR/server.log"
    else
      python3 "$SERVER_SCRIPT" 2>&1 | tee "$LOG_DIR/server.log"
    fi
    ;;
  fg-device)
    # run device in foreground, tee to device.log
    if [ -x "$VENV_PY" ]; then
      "$VENV_PY" -m atlantico_rpi.device --connect --run-for 0 2>&1 | tee "$LOG_DIR/device.log"
    else
      python3 -m atlantico_rpi.device --connect --run-for 0 2>&1 | tee "$LOG_DIR/device.log"
    fi
    ;;
  fg)
    echo "Run foreground server: ./run_demo.sh fg-server in one terminal"
    echo "Run foreground device: ./run_demo.sh fg-device in another terminal"
    ;;
  *) echo "Usage: $0 start|stop|status|logs|fg|fg-server|fg-device"; exit 2 ;;
esac

