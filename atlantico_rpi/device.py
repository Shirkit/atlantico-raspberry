"""High-level device entrypoints (setup/loop) for Raspberry Pi.

This module provides minimal wiring between the MQTT background callbacks
and the main-thread work loop. It keeps the callbacks lightweight by
enqueuing events into an `EventQueue` and processing them in `loop()`.
"""

import logging
import os
import time
import uuid
from typing import Optional
import json
import struct
import sys

from .config import *
from .mqtt_client import MQTTClient
from .model_util import ModelUtil, ModelConfig, Model
from .events import EventQueue

# initialize logging: file + console. Use env var ATLANTICO_DEVICE_LOG to override path.
LOG_PATH = os.environ.get("ATLANTICO_DEVICE_LOG", os.path.join("run", "logs", "device.log"))
try:
    os.makedirs(os.path.dirname(LOG_PATH) or ".", exist_ok=True)
except Exception:
    # Best effort; if creation fails, fallback to current directory
    LOG_PATH = os.path.join('.', 'device.log')

# Configure root logger only once to avoid duplicated handlers on reloads
root_logger = logging.getLogger()
fmt = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
# Decide whether to create a dedicated device file handler. By default we
# avoid creating a file handler so test runs (pytest) that configure a
# test-scoped log file (tests.log) are the only destination. To force
# creation of the device file handler (for standalone runs), set
# ATLANTICO_DEVICE_CREATE_FILE=1 in the environment.
abs_log_path = os.path.abspath(LOG_PATH)
create_file = os.environ.get('ATLANTICO_DEVICE_CREATE_FILE', '0') == '1'
if create_file:
    has_any_file_handler = any(isinstance(h, logging.FileHandler) for h in root_logger.handlers)
    if not has_any_file_handler:
        try:
            fh = logging.FileHandler(LOG_PATH)
            fh.setFormatter(fmt)
            fh.setLevel(logging.INFO)
            root_logger.addHandler(fh)
        except Exception:
            # best-effort: if file handler cannot be created, continue without it
            pass

# Ensure log level and at least a console StreamHandler exist so test runs
# output INFO logs when no other handlers were configured by the test harness.
root_logger.setLevel(logging.INFO)
has_stream = any(isinstance(h, logging.StreamHandler) for h in root_logger.handlers)
if not has_stream:
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    ch.setLevel(logging.INFO)
    root_logger.addHandler(ch)

_LOG = logging.getLogger(__name__)
_LOG.info("Logging initialized (file=%s)", LOG_PATH)

# module-level singletons created by `setup()` (kept simple for tests)
_EVENT_QUEUE: Optional[EventQueue] = None
_MQTT_CLIENT: Optional[MQTTClient] = None
_MODEL_UTIL: Optional[ModelUtil] = None
_RAW_MODEL_DIR = "./models/raw"
_MODEL_STORE_PATH = "./models/latest_model.json"
# Federation state constants
FEDERATE_NONE = "NONE"
FEDERATE_SUBSCRIBED = "SUBSCRIBED"
FEDERATE_STARTING = "STARTING"
FEDERATE_TRAINING = "TRAINING"
FEDERATE_DONE = "DONE"

MODEL_IDLE = "IDLE"
MODEL_WAITING_DOWNLOAD = "WAITING_DOWNLOAD"
MODEL_READY_TO_TRAIN = "READY_TO_TRAIN"
MODEL_BUSY = "MODEL_BUSY"
MODEL_DONE_TRAINING = "DONE_TRAINING"

# runtime federate state
_federate_state = FEDERATE_NONE
_current_round = -1
_new_model_state = MODEL_IDLE
_current_model_metrics = None
_federate_model_config = None



def setup(connect: bool = False, mqtt_broker: Optional[str] = None, model_store_path: str = _MODEL_STORE_PATH, device_name: Optional[str] = None):
    """Initialize device runtime objects.

    Args:
        connect: if True attempt to connect to the MQTT broker immediately.
        mqtt_broker: optional broker address to pass to the client connect call.
        model_store_path: path where JSON models (model.json) will be saved.

    Returns: tuple(EventQueue, MQTTClient, ModelUtil)
    """
    global _EVENT_QUEUE, _MQTT_CLIENT, _MODEL_UTIL, _MODEL_STORE_PATH

    _EVENT_QUEUE = EventQueue()
    # If caller provided a device_name prefer it when constructing the MQTT client
    if device_name:
        _MQTT_CLIENT = MQTTClient(client_id=device_name)
    else:
        _MQTT_CLIENT = MQTTClient()
    _MODEL_STORE_PATH = model_store_path

    # register default callbacks so network thread only enqueues light events
    _MQTT_CLIENT.register_default_handlers(_EVENT_QUEUE)

    # create a simple ModelUtil with a conservative config; TF-specific
    # behaviour will be implemented later in ModelUtil.
    cfg = ModelConfig(layers=[10, 10], activation_functions=[0, 0], epochs=1)
    _MODEL_UTIL = ModelUtil(cfg)

    # attempt to restore previous runtime config (round/federate state)
    try:
        if load_device_config():
            _LOG.info('Loaded device configuration from disk')
            # if we were training previously, notify server we can resume
            if _federate_state != FEDERATE_NONE and _current_round != -1:
                try:
                    send_command('resume')
                except Exception:
                    _LOG.debug('Failed to send resume command during setup')
    except Exception:
        _LOG.exception('Error loading device config during setup')

    # ensure model directories exist
    rawdir = os.path.dirname(_RAW_MODEL_DIR)
    try:
        os.makedirs(_RAW_MODEL_DIR, exist_ok=True)
        os.makedirs(os.path.dirname(_MODEL_STORE_PATH) or '.', exist_ok=True)
    except Exception:
        _LOG.exception("Failed to create model directories")

    if connect:
        try:
            # only pass host if explicitly provided, otherwise let MQTTClient use its default
            if mqtt_broker:
                _MQTT_CLIENT.connect(host=mqtt_broker)
            else:
                _MQTT_CLIENT.connect()
            _MQTT_CLIENT.loop_start()
        except Exception:
            _LOG.exception("Failed to connect to MQTT broker during setup")

    return _EVENT_QUEUE, _MQTT_CLIENT, _MODEL_UTIL


def loop(timeout: float = 0.1) -> None:
    """Main loop: process a single event from queue (non-blocking) and return.

    This function is intentionally small so callers can call it frequently.
    It pulls a single event (if any) and performs a light action: save model
    files, call ModelUtil stubs (train/predict) and publish minimal acks.
    """
    global _EVENT_QUEUE, _MQTT_CLIENT, _MODEL_UTIL, _federate_state, _current_round, _federate_model_config, _current_model_metrics

    if _EVENT_QUEUE is None:
        raise RuntimeError("Device not initialized; call setup() first")

    ev = _EVENT_QUEUE.try_get()
    if ev is None:
        # nothing to do
        time.sleep(timeout)
        return

    _LOG.info("Handling event %s", ev.name)

    try:
        if ev.name.startswith("command."):
            # command payload expected to be a dict that includes 'command' etc.
            _LOG.info("Received command %s payload=%s", ev.name, ev.payload)
            try:
                data = ev.payload if isinstance(ev.payload, dict) else {}
                cmd = data.get('command') or ev.name.split('.', 1)[1]
                # join -> subscribe to federation
                if cmd == 'join':
                    if _federate_state == FEDERATE_NONE:
                        _federate_state = FEDERATE_SUBSCRIBED
                        save_device_config()
                        send_command('join')
                elif cmd == 'federate_join':
                    if _federate_state == FEDERATE_NONE:
                        _federate_state = FEDERATE_SUBSCRIBED
                        save_device_config()
                        send_command('join')
                elif cmd == 'federate_unsubscribe' or cmd == 'leave':
                    _federate_state = FEDERATE_NONE
                    _current_round = -1
                    save_device_config()
                    send_command('leave')
                elif cmd == 'federate_start' or cmd == 'start':
                    # load configuration if present
                    cfg = data.get('config')
                    if cfg:
                        _federate_model_config = cfg
                    _federate_state = FEDERATE_TRAINING
                    _current_round = 0
                    save_device_config()
                    send_command('federate_start')
                elif cmd == 'request_model' or cmd == 'request':
                    # publish current model
                    try:
                        cur = None
                        if _MODEL_UTIL is not None and os.path.exists(_MODEL_STORE_PATH):
                            cur = _MODEL_UTIL.load_model_from_disk(_MODEL_STORE_PATH)
                        send_model_to_network(cur, _current_model_metrics, raw_model_path=None)
                    except Exception:
                        _LOG.exception('Failed to respond to request_model')
                elif cmd == 'resume' or cmd == 'federate_resume':
                    # mark resume requested; server should send raw model next
                    send_command('resume')
                else:
                    # best-effort ack
                    if _MQTT_CLIENT is not None:
                        try:
                            topic = 'atlantico/ack'
                            payload = json.dumps({'command': cmd, 'status': 'ok'})
                            _MQTT_CLIENT.publish(topic, payload)
                        except Exception:
                            _LOG.debug('Failed to publish ack for command')
                
            except Exception:
                _LOG.exception('Error handling command')

        elif ev.name == "model.raw":
            # save raw model bytes to a file for later processing
            data = ev.payload or {}
            payload = data.get("payload") if isinstance(data, dict) else None
            if payload:
                fname = f"{int(time.time())}-{uuid.uuid4().hex}.bin"
                path = os.path.join(_RAW_MODEL_DIR, fname)
                try:
                    # ensure destination directory exists (tests may override _RAW_MODEL_DIR)
                    dst_dir = os.path.dirname(path)
                    if dst_dir and not os.path.exists(dst_dir):
                        os.makedirs(dst_dir, exist_ok=True)

                    with open(path, "wb") as f:
                        f.write(payload)
                    _LOG.info("Saved raw model to %s", path)
                    # trigger processing: transform bytes -> Model and handle
                    try:
                        if _MODEL_UTIL is None:
                            _LOG.warning('No ModelUtil configured; skipping raw model processing')
                        else:
                            m = _MODEL_UTIL.transform_data_to_model(payload)
                            # persist JSON model derived from raw under the raw models folder
                            # to avoid clobbering the canonical latest_model.json which may
                            # have been provided as a JSON model earlier.
                            raw_json_dir = os.path.dirname(_RAW_MODEL_DIR) or '.'
                            json_path = os.path.join(raw_json_dir, f"{fname}.json")
                            _MODEL_UTIL.save_model_to_disk(m, json_path)
                            # simulate training/publish path
                            metrics = None
                            try:
                                metrics = _MODEL_UTIL.train_model_from_original_dataset(m, X_TRAIN_PATH, Y_TRAIN_PATH)
                            except Exception:
                                _LOG.debug('Training skipped or failed (TF may be missing)')
                            # publish metrics and raw model to network
                            send_model_to_network(m, metrics, raw_model_path=path)
                            # decide whether to adopt the new model
                            try:
                                if metrics is None:
                                    _LOG.info('Raw model received but no training metrics available; not adopting automatically')
                                else:
                                    if compare_metrics(_current_model_metrics, metrics):
                                        # accept new model
                                        _current_model_metrics = metrics
                                        _MODEL_UTIL.save_model_to_disk(m, _MODEL_STORE_PATH)
                                        save_device_config()
                                        _LOG.info('New model accepted and stored as current model')
                                    else:
                                        _LOG.info('New model rejected based on metrics comparison')
                            except Exception:
                                _LOG.exception('Error while comparing/persisting new model')
                    except Exception:
                        _LOG.exception('Failed to process raw model')
                except Exception:
                    _LOG.exception("Failed to save raw model")

        elif ev.name == "model.json":
            # payload is expected to be a dict with model arrays; persist using ModelUtil
            if _MODEL_UTIL is None:
                _LOG.warning("No ModelUtil configured; skipping model.json event")
            else:
                try:
                    # attempt to coerce payload -> Model and save
                    payload = ev.payload if isinstance(ev.payload, dict) else {}
                    m = Model()
                    m.biases = payload.get("biases", [])
                    m.weights = payload.get("weights", [])
                    m.parsing_time = int(payload.get("parsing_time", 0))
                    m.round = int(payload.get("round", -1))
                    _MODEL_UTIL.save_model_to_disk(m, _MODEL_STORE_PATH)
                    _LOG.info("Saved JSON model to %s", _MODEL_STORE_PATH)
                    # optionally train and publish
                    metrics = None
                    try:
                        metrics = _MODEL_UTIL.train_model_from_original_dataset(m, X_TRAIN_PATH, Y_TRAIN_PATH)
                    except Exception:
                        _LOG.debug('Training skipped or failed (TF may be missing)')
                    # save a raw representation as well for compatibility (serialize weights to bytes)
                    try:
                        raw_path = os.path.join(_RAW_MODEL_DIR, f"{int(time.time())}-{uuid.uuid4().hex}.bin")
                        os.makedirs(os.path.dirname(raw_path), exist_ok=True)
                        # naive binary: write biases then weights as float32
                        with open(raw_path, 'wb') as rf:
                            for b in (m.biases or []):
                                rf.write(struct.pack('<f', float(b)))
                            for w in (m.weights or []):
                                rf.write(struct.pack('<f', float(w)))
                        # publish metadata and raw
                        send_model_to_network(m, metrics, raw_model_path=raw_path)
                        # compare and possibly accept the new model
                        try:
                            if compare_metrics(_current_model_metrics, metrics):
                                _current_model_metrics = metrics
                                _MODEL_UTIL.save_model_to_disk(m, _MODEL_STORE_PATH)
                                save_device_config()
                                _LOG.info('New JSON model accepted and stored as current model')
                            else:
                                _LOG.info('New JSON model rejected based on metrics comparison')
                        except Exception:
                            _LOG.exception('Error comparing/persisting JSON model')
                    except Exception:
                        _LOG.exception('Failed to write/send raw model from JSON payload')
                except Exception:
                    _LOG.exception("Failed to persist JSON model")

        else:
            _LOG.debug("Unhandled event %s", ev.name)

    finally:
        try:
            _EVENT_QUEUE.task_done()
        except Exception:
            pass


def send_command(command: str, extra: dict | None = None) -> None:
    """Send a federate command to the server via MQTT_SEND_COMMANDS_TOPIC."""
    global _MQTT_CLIENT
    try:
        payload = {'command': command, 'client': getattr(_MQTT_CLIENT, 'client_id', 'atlantico-pi')}
        if extra:
            payload.update(extra)
        topic = MQTT_SEND_COMMANDS_TOPIC
        if _MQTT_CLIENT is not None:
            _MQTT_CLIENT.publish(topic, json.dumps(payload))
        else:
            _LOG.debug('MQTT client not initialized; skipping send_command')
    except Exception:
        _LOG.exception('Failed to send command')


def send_model_to_network(model: Optional[Model], metrics: object, raw_model_path: Optional[str] = None) -> None:
    """Publish JSON metrics and optionally a raw binary model to MQTT topics.

    Publishes JSON metadata to `MQTT_PUBLISH_TOPIC` and, if `raw_model_path` is
    provided, publishes the raw bytes to `MQTT_RAW_PUBLISH_TOPIC/<client>`.
    """
    global _MQTT_CLIENT
    try:
        payload = {
            'client': getattr(_MQTT_CLIENT, 'client_id', 'atlantico-pi'),
            'metrics': {},
            'model': [],
            'timings': {},
            'epochs': getattr(getattr(metrics, 'epochs', None), '__int__', lambda: 0)() if metrics is not None else 0,
        }
        # try to include some metrics fields if present
        if metrics is not None:
            for key in ('mean_squared_error', 'meanSqrdError', 'accuracy', 'precision', 'recall', 'f1Score', 'training_time', 'parsing_time'):
                if hasattr(metrics, key):
                    payload['metrics'][key] = getattr(metrics, key)
                elif isinstance(metrics, dict) and key in metrics:
                    payload['metrics'][key] = metrics[key]

        # publish JSON metadata
        try:
            topic = MQTT_PUBLISH_TOPIC
            if _MQTT_CLIENT is not None:
                _MQTT_CLIENT.publish(topic, json.dumps({'data': payload}))
            else:
                _LOG.debug('MQTT client not initialized; skipping JSON publish')
        except Exception:
            _LOG.exception('Failed to publish JSON model metadata')

        # publish raw model bytes if available
        if raw_model_path and os.path.exists(raw_model_path):
            try:
                with open(raw_model_path, 'rb') as f:
                    data = f.read()
                # Publish raw bytes to the canonical raw publish topic (no per-client suffix)
                # to keep behavior identical to the ESP32 firmware.
                topic = MQTT_RAW_PUBLISH_TOPIC
                if _MQTT_CLIENT is not None:
                    _MQTT_CLIENT.publish(topic, data)
                else:
                    _LOG.debug('MQTT client not initialized; skipping raw publish')
            except Exception:
                _LOG.exception('Failed to publish raw model bytes')
    except Exception:
        _LOG.exception('send_model_to_network failed')


def save_device_config(path: str = CONFIGURATION_PATH) -> bool:
    """Persist minimal device configuration (round, federate state, model state, metrics) to JSON."""
    global _federate_state, _current_round, _new_model_state, _current_model_metrics, _federate_model_config
    try:
        payload = {
            'currentRound': _current_round,
            'federateState': _federate_state,
            'modelState': _new_model_state,
            'metrics': {},
        }
        if _current_model_metrics is not None:
            # try dict-like or object attributes
            if isinstance(_current_model_metrics, dict):
                payload['metrics'] = _current_model_metrics
            else:
                for key in ('accuracy', 'precision', 'recall', 'f1Score', 'mean_squared_error', 'training_time', 'parsing_time'):
                    if hasattr(_current_model_metrics, key):
                        payload['metrics'][key] = getattr(_current_model_metrics, key)

        if _federate_model_config is not None:
            payload['federateModelConfig'] = _federate_model_config

        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(payload, f)
        return True
    except Exception:
        _LOG.exception('Failed to save device config')
        return False


def load_device_config(path: str = CONFIGURATION_PATH) -> bool:
    """Load device config if present and populate runtime state."""
    global _federate_state, _current_round, _new_model_state, _current_model_metrics, _federate_model_config
    try:
        if not os.path.exists(path):
            return False
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        _current_round = data.get('currentRound', -1)
        _federate_state = data.get('federateState', FEDERATE_NONE)
        _new_model_state = data.get('modelState', MODEL_IDLE)
        metrics = data.get('metrics', {})
        _current_model_metrics = metrics
        _federate_model_config = data.get('federateModelConfig')
        return True
    except Exception:
        _LOG.exception('Failed to load device config')
        return False


def compare_metrics(old_metrics, new_metrics) -> bool:
    """Decide if new_metrics are better than old_metrics using simple heuristics.

    If metrics are dicts, try keys 'accuracy','precision','recall','f1Score','meanSqrdError' (lower is better for MSE).
    Returns True if new_metrics looks better.
    """
    try:
        if old_metrics is None:
            return True
        def get(m, k):
            if m is None:
                return None
            if isinstance(m, dict):
                return m.get(k)
            return getattr(m, k, None)

        # prefer higher accuracy/f1 and lower mean squared error
        new_acc = get(new_metrics, 'accuracy') or get(new_metrics, 'acc')
        old_acc = get(old_metrics, 'accuracy') or get(old_metrics, 'acc')
        new_f1 = get(new_metrics, 'f1Score') or get(new_metrics, 'f1')
        old_f1 = get(old_metrics, 'f1Score') or get(old_metrics, 'f1')
        new_mse = get(new_metrics, 'meanSqrdError') or get(new_metrics, 'mean_squared_error')
        old_mse = get(old_metrics, 'meanSqrdError') or get(old_metrics, 'mean_squared_error')

        score = 0
        if new_acc is not None and old_acc is not None:
            if float(new_acc) > float(old_acc):
                score += 1
            elif float(new_acc) < float(old_acc):
                score -= 1
        if new_f1 is not None and old_f1 is not None:
            if float(new_f1) > float(old_f1):
                score += 1
            elif float(new_f1) < float(old_f1):
                score -= 1
        if new_mse is not None and old_mse is not None:
            if float(new_mse) < float(old_mse):
                score += 1
            elif float(new_mse) > float(old_mse):
                score -= 1

        return score >= 0
    except Exception:
        _LOG.exception('compare_metrics failure')
        return False



def main():
    """Entry point when running as a process.

    Expected behavior:
    - call setup()
    - enter a loop that calls loop() periodically
    - handle clean shutdown on SIGINT/SIGTERM
    """
    import argparse
    import signal
    import sys

    parser = argparse.ArgumentParser(description="Atlantico device runtime")
    parser.add_argument("--connect", action="store_true", help="Connect to MQTT broker during setup")
    parser.add_argument("--broker", default=None, help="MQTT broker address (overrides config)")
    parser.add_argument("--device-name", "-n", default=None, help="Override the MQTT client/device name")
    parser.add_argument("--run-for", type=float, default=0.0, help="Run for N seconds then exit (0 = forever)")
    args = parser.parse_args()

    print("Starting Atlantico device (Raspberry Pi)")
    try:
        setup(connect=args.connect, mqtt_broker=args.broker, device_name=args.device_name)
    except Exception as e:
        print("Failed to initialize device:", e)
        return

    stop_requested = False

    def _handle_signals(signum, frame):
        nonlocal stop_requested
        _LOG.info("Received signal %s, stopping", signum)
        stop_requested = True

    signal.signal(signal.SIGINT, _handle_signals)
    signal.signal(signal.SIGTERM, _handle_signals)

    start = time.time()
    try:
        while not stop_requested:
            loop()
            if args.run_for > 0 and (time.time() - start) >= args.run_for:
                _LOG.info("Run-time limit reached, exiting main loop")
                break
    finally:
        # attempt a clean shutdown
        try:
            if _MQTT_CLIENT:
                _MQTT_CLIENT.loop_stop()
        except Exception:
            _LOG.exception("Error during shutdown")
        print("Shutting down")


if __name__ == "__main__":
    main()
