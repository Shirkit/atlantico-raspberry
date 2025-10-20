# Atlantico Raspberry Pi port

This folder contains a Raspberry Pi (Zero 2 W) port of the Atlantico device code.
The goal is to replicate the high-level behavior of the ESP32 firmware in Python so you
can run development and testing on a Raspberry Pi.

## Structure

- `atlantico_rpi/` - the Python module with the core code:
  - `mqtt_client.py` - MQTT client wrapper (paho-mqtt)
  - `model_util.py` - model utilities (save/load, stubs for training/transform)
  - `events.py` - a tiny EventQueue helper for signaling heavy work to the main thread
  - `device.py` - high-level `setup()` / `loop()` placeholders
  - `config.py` - MQTT topics and file paths
- `tests/` - unit and integration tests (pytest)
- `device.json` - example device definition used to derive the MQTT client id

## Quick setup

1. Create and activate a Python virtual environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

If you don't have a `requirements.txt` yet, at minimum install:

```bash
pip install pytest paho-mqtt
```

## Tests

Run unit tests with:

```bash
pytest -q tests
```

### Development environment (recommended)

On modern systems the project runs best with Python 3.11. If your default
`python` is newer (for example Python 3.14) you can create a venv using a
system Python 3.11 or use `pyenv` to install one:

```bash
# using system python3.11
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

The integration test (`tests/test_mqtt_integration.py`) will attempt to connect to the broker
configured in `atlantico_rpi/config.py`. If the broker is not reachable the test will be skipped.

## device.json

The MQTT client will attempt to load `device.json` from the project root to derive the
`client_id` (looks for keys `client_id`, `client_name`, `device_id`, `name`). A sample
`device.json` is included.

## MQTT behavior

- Topics mirror the ESP32 implementation and are defined in `config.py`.
- Metrics are sent/received as JSON. Models are transferred as binary payloads.
- The MQTT client runs paho's network loop in a background thread and dispatches
  incoming messages to registered callbacks. Heavy work should be signaled via the
  `EventQueue` rather than done directly in the callback.

## Next steps

1. Implement `ModelUtil.transform_data_to_model` and training/prediction using TensorFlow.
2. Implement the device `setup()` and `loop()` to orchestrate training and networking.
3. Wire MQTT topic handlers to the `EventQueue` and implement the main thread worker.

If you want, I can: add a `requirements.txt` pinning `paho-mqtt>=1.6`, implement the
topic handlers now, or convert the MQTT client to use the v2 callback API explicitly.
