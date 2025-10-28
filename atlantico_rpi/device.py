"""Device entrypoints: setup, loop and background model processing.

Lightweight glue between MQTT callbacks (which enqueue events) and the
main-thread event loop that performs small actions and defers heavy work
to a background worker.
"""

import logging
import os
import time
import uuid
from typing import Optional
import json
import struct
import sys
import threading

from .config import *
from .mqtt_client import MQTTClient
from .model_util import ModelUtil, ModelConfig, Model
from .events import EventQueue
from .logging import setup_logging, LOG_PATH

setup_logging()

_LOG = logging.getLogger(__name__)
_LOG.info("Logging initialized (file=%s)", LOG_PATH)

# module-level singletons created by `setup()`
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

_federate_state = FEDERATE_NONE
_current_round = -1
_new_model_state = MODEL_IDLE
_current_model_metrics = None
_federate_model_config = None
_PROCESS_THREAD_STARTED = False
_PROCESS_LOCK = threading.Lock()



def setup(connect: bool = False, mqtt_broker: Optional[str] = None, model_store_path: str = _MODEL_STORE_PATH, device_name: Optional[str] = None):
    """Initialize runtime: EventQueue, MQTT client and ModelUtil.

    Returns (EventQueue, MQTTClient, ModelUtil). Keep this function fast.
    """
    global _EVENT_QUEUE, _MQTT_CLIENT, _MODEL_UTIL, _MODEL_STORE_PATH
    _EVENT_QUEUE = EventQueue()
    _MQTT_CLIENT = MQTTClient(client_id=device_name) if device_name else MQTTClient()
    _MODEL_STORE_PATH = model_store_path

    _MQTT_CLIENT.register_default_handlers(_EVENT_QUEUE)

    cfg = ModelConfig(layers=[10, 10], activation_functions=[0, 0], epochs=1)
    _MODEL_UTIL = ModelUtil(cfg)

    if load_device_config():
        _LOG.info('Loaded device configuration from disk')
        if _federate_state != FEDERATE_NONE and _current_round != -1:
            send_command('resume')

    os.makedirs(_RAW_MODEL_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(_MODEL_STORE_PATH) or '.', exist_ok=True)

    if connect:
        if mqtt_broker:
            _MQTT_CLIENT.connect(host=mqtt_broker)
        else:
            _MQTT_CLIENT.connect()
        _MQTT_CLIENT.loop_start()

    global _PROCESS_THREAD_STARTED
    if not _PROCESS_THREAD_STARTED:
        t = threading.Thread(target=_process_model_worker, daemon=True)
        t.start()
        _PROCESS_THREAD_STARTED = True
        _LOG.info('Started process_model background worker')

    return _EVENT_QUEUE, _MQTT_CLIENT, _MODEL_UTIL


def start_training_from_config(cfg_local):
    """Apply federate model config, persist it and mark READY_TO_TRAIN.

    Fast-path only: heavy work (training/serialization) runs in the
    background worker when the state is READY_TO_TRAIN.
    """
    global _MODEL_UTIL, _federate_model_config, _new_model_state, _federate_state, _current_round
    try:
        if isinstance(cfg_local, dict) and cfg_local.get('sendJsonWeights') is True:
            _LOG.info('start_training_from_config: sendJsonWeights requested; not supported — leaving federation')
            try:
                _federate_state = FEDERATE_NONE
                _current_round = -1
                save_device_config()
                try:
                    send_command('leave')
                except Exception:
                    _LOG.debug('Failed to send leave command')
            except Exception:
                _LOG.exception('Error while aborting federation for sendJsonWeights')
            return False
        # Guard against duplicate starts.
        if _new_model_state in (MODEL_BUSY, MODEL_READY_TO_TRAIN):
            _LOG.info('start_training_from_config: model already busy or ready (state=%s); ignoring', _new_model_state)
            return False
        # If config is provided, update ModelUtil so training uses correct arch.
        if cfg_local and isinstance(cfg_local, dict):
            try:
                existing_cfg = getattr(_MODEL_UTIL, 'config', None)
                mc = ModelConfig(
                    layers=cfg_local.get('layers', getattr(existing_cfg, 'layers', [10, 10])),
                    activation_functions=cfg_local.get('actvFunctions', cfg_local.get('activation_functions', getattr(existing_cfg, 'activation_functions', [0, 0]))),
                    epochs=cfg_local.get('epochs', getattr(existing_cfg, 'epochs', 1)),
                    random_seed=cfg_local.get('randomSeed', getattr(existing_cfg, 'random_seed', 10)),
                    learning_rate_of_weights=cfg_local.get('learningRateOfWeights', getattr(existing_cfg, 'learning_rate_of_weights', 0.3333)),
                    learning_rate_of_biases=cfg_local.get('learningRateOfBiases', getattr(existing_cfg, 'learning_rate_of_biases', 0.0666)),
                    json_weights=cfg_local.get('jsonWeights', getattr(existing_cfg, 'json_weights', False)),
                )
                _MODEL_UTIL = ModelUtil(mc)
            except Exception:
                _LOG.exception('Failed to apply federate model config; using existing ModelUtil config')

            # remember federate config for diagnostics
            _federate_model_config = cfg_local

    # Mark model ready; background worker will perform heavy work.
        _new_model_state = MODEL_READY_TO_TRAIN
        save_device_config()
        _LOG.info('Federate setup complete; model marked READY_TO_TRAIN')
        return True
    except Exception:
        _LOG.exception('Failed during federated setup')
        return False


def loop(timeout: float = 0.1) -> None:
    """Process one queued event (non-blocking) and return quickly."""
    global _EVENT_QUEUE, _MQTT_CLIENT, _MODEL_UTIL, _federate_state, _current_round, _federate_model_config, _current_model_metrics

    if _EVENT_QUEUE is None:
        raise RuntimeError('Device not initialized; call setup() first')

    ev = _EVENT_QUEUE.try_get()
    if ev is None:
        # nothing to do
        time.sleep(timeout)
        return

    _LOG.info('Handling event %s', ev.name)

    try:
        # only handle command.* events here; others are ignored quickly
        if not ev.name.startswith('command.'):
            _LOG.debug('Unhandled event %s', ev.name)
            return

        _LOG.info('Received command %s payload=%s', ev.name, ev.payload)
        data = ev.payload if isinstance(ev.payload, dict) else {}
        cmd = data.get('command') or ev.name.split('.', 1)[1]

        if cmd in ('join', 'federate_join'):
            if _federate_state == FEDERATE_NONE:
                _federate_state = FEDERATE_SUBSCRIBED
                save_device_config()
                send_command('join')
            return

        if cmd in ('federate_unsubscribe', 'leave'):
            _federate_state = FEDERATE_NONE
            _current_round = -1
            save_device_config()
            send_command('leave')
            return

        if cmd in ('federate_start', 'start'):
            cfg = data.get('config')
            if cfg:
                _federate_model_config = cfg
            _federate_state = FEDERATE_TRAINING
            _current_round = 0
            save_device_config()
            send_command('start')
            try:
                start_training_from_config(cfg)
            except Exception:
                _LOG.exception('Failed to start federated training setup')
            return

        if cmd in ('request_model', 'request'):
            cur = None
            if _MODEL_UTIL is not None and os.path.exists(_MODEL_STORE_PATH):
                cur = _MODEL_UTIL.load_model_from_disk(_MODEL_STORE_PATH)
            send_model_to_network(cur, _current_model_metrics, raw_model_path=None)
            return

        if cmd in ('resume', 'federate_resume'):
            send_command('resume')
            return

        if cmd in ('alive', 'federate_alive'):
            send_command('alive')
            return

        _LOG.debug('Unknown command %s', cmd)
    finally:
        try:
            _EVENT_QUEUE.task_done()
        except Exception:
            _LOG.debug('EventQueue.task_done failed', exc_info=True)


def _process_model_worker():
    """Daemon worker: when state==READY_TO_TRAIN it trains and serializes.

    Keeps MQTT thread responsive by doing heavy work off the callback thread.
    """
    global _new_model_state, _current_model_metrics, _MODEL_UTIL, _federate_state, _current_round
    _LOG.info('process_model worker entering loop')
    while True:
        try:
            if _new_model_state != MODEL_READY_TO_TRAIN:
                time.sleep(0.5)
                continue

            with _PROCESS_LOCK:
                if _new_model_state != MODEL_READY_TO_TRAIN:
                    continue
                _LOG.info('process_model: detected READY_TO_TRAIN -> starting')
                _new_model_state = MODEL_BUSY
                save_device_config()

            if _MODEL_UTIL is None:
                _LOG.warning('process_model: no ModelUtil configured; skipping training')
                _new_model_state = MODEL_IDLE
                save_device_config()
                time.sleep(0.5)
                continue

            metrics = _MODEL_UTIL.train_model_from_original_dataset(Model(), X_TRAIN_PATH, Y_TRAIN_PATH)
            if metrics is None:
                _LOG.warning('process_model: training returned no metrics')
                _new_model_state = MODEL_IDLE
                save_device_config()
                time.sleep(0.5)
                continue

            _current_model_metrics = metrics
            _new_model_state = MODEL_DONE_TRAINING
            save_device_config()
            _LOG.info('process_model: training complete (metrics=%s)', getattr(metrics, '__dict__', metrics))

            raw_path = None
            tf_model = getattr(_MODEL_UTIL, '_last_trained_tf_model', None)
            if tf_model is not None:
                raw_bytes = _MODEL_UTIL.serialize_to_nn_bytes(keras_model=tf_model)
                if raw_bytes:
                    os.makedirs(_RAW_MODEL_DIR, exist_ok=True)
                    raw_path = os.path.join(_RAW_MODEL_DIR, f"{int(time.time())}-{uuid.uuid4().hex}.nn")
                    with open(raw_path, 'wb') as rf:
                        rf.write(raw_bytes)
                    _LOG.info('process_model: wrote NN binary to %s', raw_path)

            try:
                send_model_to_network(None, metrics, raw_model_path=raw_path)
                _LOG.info('process_model: sent model to network')
            except Exception:
                _LOG.exception('process_model: failed to send model to network')

            _new_model_state = MODEL_IDLE
            save_device_config()
            _LOG.info('process_model: state set to IDLE')
            time.sleep(0.5)
        except Exception:
            _LOG.exception('process_model worker error')
            time.sleep(1.0)


def send_command(command: str, extra: dict | None = None) -> None:
    """Publish a federate command to MQTT_SEND_COMMANDS_TOPIC."""
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


def send_model_to_network(model: Optional[Model], metrics: object, raw_model_bytes: Optional[bytes] = None, raw_model_path: Optional[str] = None) -> None:
    """Publish metrics JSON to MQTT_PUBLISH_TOPIC and raw bytes to raw topic.

    JSON contains client, metrics and epochs. Raw bytes are sent to
    MQTT_RAW_PUBLISH_TOPIC/<client> when available.
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

        # publish JSON metadata (metrics only)
        try:
            # Publish only metrics and client info on the JSON publish topic.
            topic = MQTT_PUBLISH_TOPIC
            json_payload = {'client': payload.get('client'), 'metrics': payload.get('metrics', {}), 'epochs': payload.get('epochs', 0)}
            if _MQTT_CLIENT is not None:
                _MQTT_CLIENT.publish(topic, json.dumps({'data': json_payload}))
            else:
                _LOG.debug('MQTT client not initialized; skipping JSON publish')
        except Exception:
            _LOG.exception('Failed to publish JSON model metadata')

        # publish raw model bytes if available (prefer in-memory bytes)
        try:
            data = None
            if raw_model_bytes:
                data = raw_model_bytes
            elif model is not None and _MODEL_UTIL is not None:
                try:
                    data = _MODEL_UTIL.serialize_to_nn_bytes(model=model)
                except Exception:
                    data = None
            elif raw_model_path and os.path.exists(raw_model_path):
                try:
                    with open(raw_model_path, 'rb') as f:
                        data = f.read()
                except Exception:
                    _LOG.exception('Failed to read raw model from path')

            if data:
                topic = MQTT_RAW_PUBLISH_TOPIC
                client_suffix = getattr(_MQTT_CLIENT, 'client_id', None) or 'atlantico-pi'
                topic = f"{topic}/{client_suffix}"
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
    payload = {
        'currentRound': _current_round,
        'federateState': _federate_state,
        'modelState': _new_model_state,
        'metrics': {},
    }
    if _current_model_metrics is not None:
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
    # Ensure file+stdout handlers for standalone runs
    try:
        setup_logging(force_file=True)
    except Exception:
        _LOG.debug('setup_logging(force_file=True) failed', exc_info=True)

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
