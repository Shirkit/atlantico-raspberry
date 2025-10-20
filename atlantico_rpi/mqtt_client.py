"""MQTT client wrapper for Raspberry Pi using paho-mqtt.

Small thread-based client that dispatches incoming messages to callbacks.
"""

import logging
import threading
import time
from typing import Callable, Dict, Optional
import json
import os
import socket

import paho.mqtt.client as mqtt
from .config import MQTT_BROKER, MQTT_RECEIVE_TOPIC, MQTT_RAW_RECEIVE_TOPIC, MQTT_RECEIVE_COMMANDS_TOPIC, MQTT_RESUME_TOPIC, MQTT_RAW_RESUME_TOPIC
from .events import EventQueue

_LOG = logging.getLogger(__name__)


class MQTTClient:
    def __init__(self, client_id: Optional[str] = None, device_json_path: str = "./device.json"):
        resolved_client_id = client_id
        if not resolved_client_id:
            try:
                if os.path.exists(device_json_path):
                    with open(device_json_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        for key in ("client_id", "client_name", "device_id", "name"):
                            if key in data and isinstance(data[key], str) and data[key].strip():
                                resolved_client_id = data[key].strip()
                                break
            except Exception:
                _LOG.exception("Failed to read device.json at %s", device_json_path)

        if not resolved_client_id:
            try:
                resolved_client_id = socket.gethostname()
            except Exception:
                resolved_client_id = "atlantico-pi"

        self.client_id = resolved_client_id
        _LOG.debug("Using MQTT client id: %s", self.client_id)
        # Some paho-mqtt versions expose CallbackAPIVersion; use it if present.
        callback_api = None
        try:
            CallbackAPIVersion = getattr(mqtt, 'CallbackAPIVersion', None)
            if CallbackAPIVersion is not None:
                callback_api = getattr(CallbackAPIVersion, 'VERSION2', None)
        except Exception:
            callback_api = None

        # Construct client with callback_api_version only when available to remain
        # compatible with older paho-mqtt builds and avoid static analyzer errors.
        if callback_api is not None:
            self._client = mqtt.Client(client_id=self.client_id, callback_api_version=callback_api)
        else:
            self._client = mqtt.Client(client_id=self.client_id)
        self._connected = False
        self._callbacks: Dict[str, Callable[[str, bytes], None]] = {}
        self._lock = threading.RLock()
        self._should_stop = threading.Event()

        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._client.on_message = self._on_message
        # Enable paho debug logger to surface lower-level MQTT errors
        try:
            mqtt_logger = logging.getLogger('paho')
            mqtt_logger.setLevel(logging.INFO)
            self._client.enable_logger(mqtt_logger)
        except Exception:
            _LOG.debug("Failed to enable paho logger")

    # ------------------ Public API ------------------
    def connect(self, host: str = MQTT_BROKER, port: int = 1883, keepalive: int = 60, timeout: int = 10) -> None:
        """Connect to the MQTT broker and start the network loop.

        Blocks until the client reports it's connected or until timeout.
        """
        _LOG.info("Connecting to MQTT broker %s:%s as %s", host, port, self.client_id)
        try:
            self._client.connect(host, port, keepalive)
        except Exception as e:
            _LOG.exception("Failed to start connection attempt: %s", e)
            raise

        # start the network loop in the background so on_connect will be called
        self._client.loop_start()

        # wait until on_connect sets the flag or timeout
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._connected:
                _LOG.info("Connected to MQTT broker")
                return
            time.sleep(0.1)

        # if we reach here, connection didn't happen in time
        raise RuntimeError("Timeout while connecting to MQTT broker")

    def publish(self, topic: str, payload: bytes | str, qos: int = 0, retain: bool = False) -> None:
        """Publish bytes or string payload to a topic (default QoS=0)."""
        if isinstance(payload, str):
            payload = payload.encode("utf-8")
        _LOG.debug("Publishing to %s (%d bytes) qos=%s retain=%s", topic, len(payload), qos, retain)
        try:
            # paho returns (rc, mid)
            rc, mid = self._client.publish(topic, payload, qos=qos, retain=retain)
            if rc != mqtt.MQTT_ERR_SUCCESS:
                _LOG.warning("Publish returned rc=%s for topic=%s", rc, topic)
        except Exception:
            _LOG.exception("Publish failed for topic=%s", topic)
            raise

    def subscribe(self, topic: str, callback: Callable[[str, bytes], None], qos: int = 0) -> None:
        """Subscribe and register callback(topic, payload_bytes). Latest callback wins."""
        with self._lock:
            _LOG.info("Registering subscription for %s qos=%s (deferred until connected if necessary)", topic, qos)
            # Store callback immediately so we can re-subscribe on connect.
            self._callbacks[topic] = callback
            # If we are already connected, perform the actual paho subscribe now.
            if self._connected:
                try:
                    result = self._client.subscribe(topic, qos)
                    # Normalize result shape
                    if isinstance(result, tuple):
                        rc = result[0]
                    else:
                        rc = result
                    if rc != mqtt.MQTT_ERR_SUCCESS:
                        _LOG.warning("Subscribe returned rc=%s for topic=%s", rc, topic)
                except Exception:
                    _LOG.exception("Exception while subscribing to %s", topic)

    def register_default_handlers(self, event_queue: EventQueue) -> None:
        """Subscribe default Atlantis topics and push light events to the queue.

        The callbacks are small and enqueue either parsed JSON for command/topics
        or raw bytes for model payloads.
        """

        def make_json_callback(event_name_prefix: str):
            def _cb(topic: str, payload: bytes):
                try:
                    data = json.loads(payload.decode("utf-8"))
                except Exception:
                    _LOG.exception("Failed to decode JSON on %s", topic)
                    return
                # command-specific event naming: e.g. 'command.join'
                name = event_name_prefix
                if isinstance(data, dict) and "command" in data:
                    name = f"command.{data['command']}"
                event_queue.put(name, data)

            return _cb

        def make_raw_callback(name: str):
            def _cb(topic: str, payload: bytes):
                event_queue.put(name, {"topic": topic, "payload": payload})

            return _cb

        # Subscribe to command topics (JSON payloads)
        self.subscribe(MQTT_RECEIVE_COMMANDS_TOPIC, make_json_callback("command"))

        # Subscribe to JSON model receives
        self.subscribe(MQTT_RECEIVE_TOPIC, make_json_callback("model.json"))

        # Subscribe to raw model receives
        self.subscribe(MQTT_RAW_RECEIVE_TOPIC, make_raw_callback("model.raw"))

        # Subscribe to resume topics
        self.subscribe(MQTT_RESUME_TOPIC, make_json_callback("command.resume"))
        self.subscribe(MQTT_RAW_RESUME_TOPIC, make_raw_callback("command.resume.raw"))

    def loop_start(self) -> None:
        """Start the paho network loop in a background thread."""
        _LOG.debug("Starting paho network loop")
        self._client.loop_start()

    def loop_stop(self, force: bool = False) -> None:
        """Stop the paho network loop and disconnect cleanly."""
        _LOG.debug("Stopping paho network loop (force=%s)", force)
        try:
            self._client.loop_stop()
        except Exception:
            _LOG.exception("Error stopping network loop")

    # ------------------ Internal callbacks ------------------
    def _on_connect(self, client, userdata, flags, rc, properties=None):
        if rc == 0:
            _LOG.info("MQTT connected")
            self._connected = True
            # re-subscribe to registered topics
            with self._lock:
                for topic in self._callbacks.keys():
                    try:
                        res = client.subscribe(topic)
                        _LOG.info("Re-subscribe result for %s -> %s", topic, res)
                    except Exception:
                        _LOG.exception("Failed to re-subscribe to %s", topic)
        else:
            _LOG.warning("MQTT on_connect rc=%s", rc)

    def _on_disconnect(self, client, userdata, rc, properties=None):
        _LOG.warning("MQTT disconnected (rc=%s)", rc)
        self._connected = False

        # if disconnection was unexpected, attempt a reconnect in background
        if rc != 0 and not self._should_stop.is_set():
            threading.Thread(target=self._reconnect_backoff, daemon=True).start()

    def _on_message(self, client, userdata, msg):
        topic = msg.topic
        payload = msg.payload
        _LOG.debug("Received message on %s (%d bytes)", topic, len(payload))

        # exact match then wildcard
        cb: Optional[Callable[[str, bytes], None]] = None
        with self._lock:
            if topic in self._callbacks:
                cb = self._callbacks[topic]
            else:
                for subscribed_topic, candidate in self._callbacks.items():
                    try:
                        if mqtt.topic_matches_sub(subscribed_topic, topic):
                            cb = candidate
                            break
                    except Exception:
                        continue

        if cb:
            try:
                cb(topic, payload)
            except Exception:
                _LOG.exception("Subscriber callback raised for %s", topic)

    def _reconnect_backoff(self):
        attempt = 0
        while not self._should_stop.is_set():
            backoff = min(30, (2 ** attempt))
            _LOG.info("Attempting MQTT reconnect (attempt=%d), sleeping %ds", attempt + 1, backoff)
            try:
                self._client.reconnect()
                _LOG.info("Reconnected to MQTT broker")
                return
            except Exception:
                _LOG.debug("Reconnect attempt failed", exc_info=True)
            time.sleep(backoff)
            attempt += 1

