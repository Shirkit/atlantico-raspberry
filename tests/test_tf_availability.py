import importlib


def test_tensorflow_or_tflite_available():
    """Passes if either tensorflow or tflite_runtime is importable in the venv."""
    tf_spec = importlib.util.find_spec('tensorflow')
    tr_spec = importlib.util.find_spec('tflite_runtime')
    assert tf_spec is not None or tr_spec is not None, "Neither tensorflow nor tflite_runtime is available in the environment"
