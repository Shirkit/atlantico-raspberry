import uuid
import queue
import time
import socket
import pytest

from atlantico_rpi.mqtt_client import MQTTClient
from atlantico_rpi.config import MQTT_BROKER


def test_mqtt_integration_publish_subscribe():
    """Integration test: connect to broker, subscribe and publish a message.

    Requires a reachable MQTT broker at `MQTT_BROKER` (no auth).
    """
    client_id = f"pi-integ-{uuid.uuid4().hex[:8]}"

    # quick reachability check for the broker so the test can be skipped in CI or
    # environments where the broker isn't reachable.
    host = MQTT_BROKER
    port = 1883
    try:
        with socket.create_connection((host, port), timeout=1):
            pass
    except Exception:
        pytest.skip(f"MQTT broker {host}:{port} not reachable from test runner")

    client = MQTTClient(client_id=client_id)
    # connect to broker (will raise if cannot connect)
    client.connect()

    q = queue.Queue()

    def on_msg(topic, payload):
        q.put((topic, payload))

    topic = f"esp32/fl/test/integration/{uuid.uuid4().hex}"
    client.subscribe(topic, on_msg)

    payload = b"integration-test-payload"
    client.publish(topic, payload)

    try:
        t, p = q.get(timeout=5)
    finally:
        # ensure we stop the loop even if test fails
        client.loop_stop()

    assert p == payload
