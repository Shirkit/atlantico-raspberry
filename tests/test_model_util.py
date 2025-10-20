from atlantico_rpi.model_util import Model, ModelConfig, ModelUtil


def test_model_save_load(tmp_path):
    m = Model(biases=[1.5, 2.5], weights=[0.1, 0.2, 0.3], parsing_time=123, round=7)
    cfg = ModelConfig([3, 4], [1, 1])
    util = ModelUtil(cfg)

    path = tmp_path / "model.json"
    ok = util.save_model_to_disk(m, str(path))
    assert ok is True

    loaded = util.load_model_from_disk(str(path))
    assert loaded.biases == m.biases
    assert loaded.weights == m.weights
    assert loaded.parsing_time == m.parsing_time
    assert loaded.round == m.round
