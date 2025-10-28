import atlantico_rpi.device as device


def test_start_training_applies_cfg_and_marks_ready():
    # backup
    orig_state = device._new_model_state
    orig_config = device._federate_model_config
    try:
        device._new_model_state = device.MODEL_IDLE
        cfg = {'layers': [5, 3], 'activation_functions': [0, 0], 'epochs': 2, 'randomSeed': 42}
        res = device.start_training_from_config(cfg)
        assert res is True
        assert device._new_model_state == device.MODEL_READY_TO_TRAIN
        assert device._federate_model_config == cfg
    finally:
        device._new_model_state = orig_state
        device._federate_model_config = orig_config


def test_start_training_guard_prevents_duplicate():
    orig_state = device._new_model_state
    orig_config = device._federate_model_config
    try:
        device._new_model_state = device.MODEL_BUSY
        cfg = {'layers': [1], 'epochs': 1}
        prev = device._federate_model_config
        res = device.start_training_from_config(cfg)
        assert res is False
        assert device._federate_model_config == prev
    finally:
        device._new_model_state = orig_state
        device._federate_model_config = orig_config
