#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEVICE_LOG="$ROOT_DIR/run/logs/device.log"
SERVER_LOG="$ROOT_DIR/../atlantico-server/run/logs/server.log"
MOSQ_LOG="$ROOT_DIR/run/logs/mosquitto.log"

usage() {
  cat <<EOF
Usage: $0 <command>

Commands:
  check       - check mosquitto process and listening port (1883)
  publish     - publish 3 test messages (commands, model, raw)
  tail        - tail server and device logs
  all         - run check, publish, tail sequentially

Examples:
  $0 check
  $0 publish
  $0 tail
  $0 all
EOF
}

check_broker() {
  echo "== Listening sockets for :1883 =="
  ss -ltnp | grep :1883 || echo "(no listener on :1883)"
  echo
  echo "== mosquitto processes =="
  ps aux | grep -E 'mosquitto' | grep -v grep || echo "(no mosquitto process)"
  echo
  if [ -f /var/log/mosquitto/mosquitto.log ]; then
    echo "== /var/log/mosquitto/mosquitto.log (tail 50) =="
    tail -n 50 /var/log/mosquitto/mosquitto.log || true
  fi
  echo
  if [ -f "$MOSQ_LOG" ]; then
    echo "== $MOSQ_LOG (tail 50) =="
    tail -n 50 "$MOSQ_LOG" || true
  fi
}

publish_tests() {
  echo "Publishing test JSON to esp32/fl/commands/pull"
  mosquitto_pub -h 127.0.0.1 -t 'esp32/fl/commands/pull' -m '{"command":"test","client":"cli-test"}' || true
  sleep 0.2

  echo "Publishing test JSON to esp32/fl/model/pull"
  mosquitto_pub -h 127.0.0.1 -t 'esp32/fl/model/pull' -m '{"client":"cli-test","model":{}}' || true
  sleep 0.2

  echo "Publishing raw test to esp32/fl/model/rawpull"
  printf 'TESTBIN' | mosquitto_pub -h 127.0.0.1 -t 'esp32/fl/model/rawpull' -s || true
  sleep 0.5
}

tail_logs() {
  echo "== Server log tail (if present) =="
  [ -f "$SERVER_LOG" ] && tail -n 200 "$SERVER_LOG" || echo "(no server log at $SERVER_LOG)"
  echo
  echo "== Device log tail (if present) =="
  [ -f "$DEVICE_LOG" ] && tail -n 200 "$DEVICE_LOG" || echo "(no device log at $DEVICE_LOG)"
}

if [ $# -lt 1 ]; then
  usage
  exit 1
fi

case "$1" in
  check) check_broker ;; 
  publish) publish_tests ;; 
  tail) tail_logs ;; 
  all) check_broker; publish_tests; tail_logs ;; 
  *) usage; exit 2 ;;
esac
