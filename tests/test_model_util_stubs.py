import pytest

from atlantico_rpi.model_util import ModelUtil, ModelConfig, Model


def test_transform_from_dict():
    cfg = ModelConfig(layers=[1], activation_functions=[0])
    mu = ModelUtil(cfg)
    payload = {"biases": [0.1], "weights": [1, 2, 3], "parsing_time": 5, "round": 2}
    m = mu.transform_data_to_model(payload)
    assert isinstance(m, Model)
    assert m.weights == [1, 2, 3]


def test_transform_from_bytes():
    cfg = ModelConfig(layers=[1], activation_functions=[0])
    mu = ModelUtil(cfg)
    raw = b"\x01\x02\x03"
    m = mu.transform_data_to_model(raw)
    assert isinstance(m, Model)
    assert isinstance(m.weights, list)


def test_train_and_predict():
    cfg = ModelConfig(layers=[1], activation_functions=[0], epochs=3)
    mu = ModelUtil(cfg)
    m = Model()
    metrics = mu.train_model_from_original_dataset(m, "x.csv", "y.csv")
    assert metrics.epochs == 3
    preds = mu.predict_from_current_model(m, [1, 2, 3])
    assert preds == [0, 0, 0]
