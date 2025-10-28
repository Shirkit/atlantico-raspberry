import json
import pytest
import atlantico_rpi.mqtt_client as mc

from atlantico_rpi.mqtt_client import MQTTClient
from atlantico_rpi.events import EventQueue


class DummyClient:
    """Minimal mock of paho Client used by MQTTClient in tests."""

    def __init__(self, *args, **kwargs):
        # accept callback_api_version and other kwargs
        self.on_connect = None
        self.on_disconnect = None
        self.on_message = None

    def subscribe(self, topic, qos=0):
        return (0, 1)


def test_register_default_handlers_enqueue_events(monkeypatch):
    # replace mqtt.Client with DummyClient so MQTTClient constructs cleanly
    monkeypatch.setattr('atlantico_rpi.mqtt_client.mqtt.Client', DummyClient)

    q = EventQueue()
    client = MQTTClient(client_id='test-client')
    client.register_default_handlers(q)

    # simulate an incoming JSON command on the commands topic
    topic = client._callbacks.keys().__iter__().__next__()
    # find the callback registered for the commands topic
    cmd_topic = None
    for t in client._callbacks:
        if 'commands' in t:
            cmd_topic = t
            break
    assert cmd_topic is not None

    cb = client._callbacks[cmd_topic]
    payload = json.dumps({"command": "join", "payload": {"round": 1}}).encode('utf-8')
    cb(cmd_topic, payload)

    ev = q.get(timeout=1.0)
    assert ev.name == 'command.join'
    assert isinstance(ev.payload, dict)
    assert ev.payload['command'] == 'join'

    # default handlers no longer subscribe to raw receive topic
    raw_cb = client._callbacks.get(mc.MQTT_RAW_RECEIVE_TOPIC)
    assert raw_cb is None
import json
from atlantico_rpi.events import EventQueue


def test_register_default_handlers(monkeypatch):
    import atlantico_rpi.mqtt_client as mc

    # Create a mock client that stores callbacks and allows simulation
    stored_callbacks = {}

    class MockClient:
        def __init__(self, client_id=None, **kwargs):
            self.on_connect = None
            self.on_disconnect = None
            self.on_message = None

        def subscribe(self, topic, qos=0):
            # pretend subscribe succeeded
            return (0, 1)

    # Patch the mqtt.Client used in the module
    monkeypatch.setattr(mc.mqtt, "Client", MockClient)

    client = mc.MQTTClient(client_id="test-client")

    # Event queue to receive events
    q = EventQueue()

    # Register default handlers (this will call subscribe and register callbacks)
    client.register_default_handlers(q)

    # Find the callback registered for the JSON command topic
    cb = client._callbacks.get(mc.MQTT_RECEIVE_COMMANDS_TOPIC)
    assert cb is not None

    # Simulate a JSON command payload
    payload = json.dumps({"command": "join", "client": "pi-test"}).encode("utf-8")
    cb(mc.MQTT_RECEIVE_COMMANDS_TOPIC, payload)

    evt = q.get(timeout=1)
    assert evt.name == "command.join"
    assert evt.payload["client"] == "pi-test"

    # The default handler should not register the raw receive callback.
    raw_cb = client._callbacks.get(mc.MQTT_RAW_RECEIVE_TOPIC)
    assert raw_cb is None
