import pytest


def test_mqttclient_mocked(monkeypatch):
    # import the module so we can patch its mqtt.Client
    import atlantico_rpi.mqtt_client as mc

    published = []
    subscribed = []

    class MockMsg:
        def __init__(self, topic, payload):
            self.topic = topic
            self.payload = payload

    class MockClient:
        def __init__(self, client_id=None, **kwargs):
            self._client_id = client_id
            self.on_connect = None
            self.on_disconnect = None
            self.on_message = None
            self._subs = []

        def connect(self, host, port, keepalive):
            # simulate immediate successful connect
            if callable(self.on_connect):
                # flags param is dict-like in paho
                self.on_connect(self, None, {}, 0)

        def loop_start(self):
            pass

        def loop_stop(self, force=False):
            pass

        def subscribe(self, topic, qos=0):
            self._subs.append((topic, qos))
            subscribed.append((topic, qos))
            return (0, 1)

        def publish(self, topic, payload, qos=0, retain=False):
            published.append((topic, payload, qos, retain))
            return (mc.mqtt.MQTT_ERR_SUCCESS, 1)

        def reconnect(self):
            if callable(self.on_connect):
                self.on_connect(self, None, {}, 0)

    # patch the Client used in the module
    monkeypatch.setattr(mc.mqtt, "Client", MockClient)

    # create MQTTClient which will use our MockClient
    client = mc.MQTTClient(client_id="test-client")

    # connect (should call mock on_connect and set connected flag)
    client.connect(timeout=1)
    assert client._connected is True

    received = []

    def cb(topic, payload):
        received.append((topic, payload))

    client.subscribe("esp32/fl/test", cb)

    # simulate an incoming message by invoking the mock on_message
    mock = client._client
    # create a message and call the assigned on_message
    msg = MockMsg("esp32/fl/test", b"hello")
    # call the on_message callback as paho would
    mock.on_message(mock, None, msg)

    assert len(received) == 1
    assert received[0][0] == "esp32/fl/test"
    assert received[0][1] == b"hello"

    # test publish
    client.publish("esp32/fl/out", b"payload")
    assert any(p[0] == "esp32/fl/out" and p[1] == b"payload" for p in published)