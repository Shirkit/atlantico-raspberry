"""Configuration constants for Raspberry Pi port"""

# Network / Broker
WIFI_SSID = "NovoOrquidario"
WIFI_PASSWORD = "orquidea"
MQTT_BROKER = "127.0.0.1"

# Paths
MODEL_PATH = "./model.nn"
NEW_MODEL_PATH = "./new_model.nn"
CONFIGURATION_PATH = "./config.json"
DEVICE_DEFINITION_PATH = "./device.json"
DATA_DIR = "./data"
X_TRAIN_PATH = "./data/x_train.csv"
Y_TRAIN_PATH = "./data/y_train.csv"
X_TEST_PATH = "./data/x_test.csv"
Y_TEST_PATH = "./data/y_test.csv"
GATHERED_DATA_PATH = "./data.db"

# MQTT
MQTT_PUBLISH_TOPIC = "esp32/fl/model/push"
MQTT_RAW_PUBLISH_TOPIC = "esp32/fl/model/rawpush"
MQTT_RECEIVE_TOPIC = "esp32/fl/model/pull"
MQTT_RAW_RECEIVE_TOPIC = "esp32/fl/model/rawpull"
MQTT_RESUME_TOPIC = "esp32/fl/model/resume"
MQTT_RAW_RESUME_TOPIC = "esp32/fl/model/rawresume"
MQTT_RECEIVE_COMMANDS_TOPIC = "esp32/fl/commands/pull"
MQTT_SEND_COMMANDS_TOPIC = "esp32/fl/commands/push"

CONNECTION_TIMEOUT = 30000  # milliseconds
