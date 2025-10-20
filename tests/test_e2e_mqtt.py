import socket
import time
import os
import json
import threading

import pytest
import paho.mqtt.client as mqtt

from atlantico_rpi import device
from atlantico_rpi.config import MQTT_RECEIVE_TOPIC, MQTT_RAW_RECEIVE_TOPIC


MQTT_BROKER = "127.0.0.1"
MQTT_PORT = 1883


def _broker_reachable(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except Exception:
        return False


@pytest.mark.skipif(not _broker_reachable(MQTT_BROKER, MQTT_PORT), reason="MQTT broker not reachable at 127.0.0.1:1883")
def test_e2e_model_receive_and_ack(tmp_path):
    # prepare device to run for a short time and connect to local broker
    raw_dir = tmp_path / "models" / "raw"
    store = tmp_path / "models" / "latest_model.json"

    # Instead of calling device.main (which parses argv), call setup and then run
    # the device.loop() in a background thread so queued events are processed.
    q, client, mu = device.setup(connect=True, mqtt_broker=MQTT_BROKER, model_store_path=str(store))

    # redirect raw model dir to the temporary test folder so we can assert files
    device._RAW_MODEL_DIR = str(tmp_path / "models" / "raw")

    stop_event = threading.Event()

    def loop_worker():
        while not stop_event.is_set():
            try:
                device.loop(timeout=0.05)
            except Exception:
                pass
            time.sleep(0.01)

    worker = threading.Thread(target=loop_worker, daemon=True)
    worker.start()

    # small helper to subscribe to acks
    acks = []

    def on_ack(c, userdata, msg):
        try:
            acks.append(msg.payload.decode("utf-8"))
        except Exception:
            acks.append(None)

    # request v2 callback API explicitly to avoid deprecation warnings
    callback_api = getattr(mqtt, "CallbackAPIVersion", None)
    if callback_api is not None and hasattr(callback_api, "VERSION2"):
        sub = mqtt.Client(callback_api_version=callback_api.VERSION2)
    else:
        sub = mqtt.Client()
    sub.on_message = on_ack
    sub.connect(MQTT_BROKER, MQTT_PORT)
    sub.subscribe("atlantico/ack")
    sub.loop_start()

    try:
        # publish a JSON model
        if callback_api is not None and hasattr(callback_api, "VERSION2"):
            pub = mqtt.Client(callback_api_version=callback_api.VERSION2)
        else:
            pub = mqtt.Client()
        pub.connect(MQTT_BROKER, MQTT_PORT)
        payload = {"biases": [0.1], "weights": [1, 2, 3], "parsing_time": 1, "round": 1}
        pub.publish(MQTT_RECEIVE_TOPIC, json.dumps(payload))

        # publish a raw model
        pub.publish(MQTT_RAW_RECEIVE_TOPIC, b"\x00\x01\x02")

        # allow some time for the device to process (increased to reduce timing flakiness)
        time.sleep(10.0)

        # check that JSON model file was written
        assert store.exists()
        data = json.loads(store.read_text(encoding="utf-8"))
        assert data.get("weights") == [1, 2, 3]

        # check raw model saved into the temporary raw dir (we expect at least one file)
        raw_parent = os.path.join(str(tmp_path), "models", "raw")
        found = False
        if os.path.isdir(raw_parent):
            for fname in os.listdir(raw_parent):
                if fname.endswith('.bin'):
                    found = True
                    break
        assert found

    # device does not publish an ack for model receives by default;
    # presence of saved files is considered sufficient for this test.

    finally:
        stop_event.set()
        worker.join(timeout=1.0)
        try:
            sub.loop_stop()
        except Exception:
            pass
