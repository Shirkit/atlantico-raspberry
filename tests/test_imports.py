import atlantico_rpi
from atlantico_rpi import device, model_util, mqtt_client, config


def test_imports():
    assert hasattr(device, 'main')
    assert hasattr(model_util, 'ModelUtil')
    assert hasattr(mqtt_client, 'MQTTClient')
    assert hasattr(config, 'MQTT_BROKER')
