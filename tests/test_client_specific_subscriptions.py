import json
import time
import types

import pytest

from atlantico_rpi.events import EventQueue
from atlantico_rpi.mqtt_client import MQTTClient
from atlantico_rpi.device import setup, _EVENT_QUEUE


class DummyMsg:
    def __init__(self, topic: str, payload: bytes):
        self.topic = topic
        self.payload = payload


def test_client_specific_raw_enqueue(tmp_path, monkeypatch):
    # create an EventQueue and a MQTTClient
    eq = EventQueue()
    client = MQTTClient(client_id="test-device")

    # register default handlers into our queue
    client.register_default_handlers(eq)

    # simulate subscribing client-specific raw topic
    topic = f"esp32/fl/model/rawresume/{client.client_id}"

    called = {'enqueued': False}

    # register a small callback that uses the default behavior
    # note: project no longer uses 'model.raw' by default; tests can use
    # a local event name to validate subscription behavior
    def cb(topic, payload):
        eq.put('model.binary', {'topic': topic, 'payload': payload})

    client.subscribe(topic, cb)

    # simulate incoming paho message
    msg = DummyMsg(topic, b"\x00\x01\x02")
    # call protected handler directly (simulates broker delivering message)
    client._on_message(None, None, msg)

    ev = eq.try_get()
    assert ev is not None
    assert ev.name == 'model.binary'
    assert isinstance(ev.payload, dict)
    assert ev.payload['topic'] == topic
    assert ev.payload['payload'] == b"\x00\x01\x02"


def test_device_setup_with_device_name(tmp_path, monkeypatch):
    # ensure setup accepts device_name and sets mqtt client id
    eq, mqtt_client, model_util = setup(connect=False, device_name="cli-device")
    assert mqtt_client.client_id == "cli-device"
    # cleanup (setup created global state)
    # no further assertions; this just validates wiring
    