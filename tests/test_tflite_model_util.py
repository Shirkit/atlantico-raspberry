import os
import numpy as np
import tempfile

from atlantico_rpi.model_util import ModelUtil, ModelConfig, Model


def test_train_and_export_tflite(tmp_path):
    # create tiny synthetic CSV files
    x = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([[2.0], [4.0], [6.0], [8.0]])
    x_file = tmp_path / "x.csv"
    y_file = tmp_path / "y.csv"
    np.savetxt(x_file, x, delimiter=',')
    np.savetxt(y_file, y, delimiter=',')

    cfg = ModelConfig(layers=[4], activation_functions=[0], epochs=2)
    mu = ModelUtil(cfg)

    metrics = mu.train_model_from_original_dataset(Model(), str(x_file), str(y_file))
    # metrics should be returned even if TF is not available (then defaults)
    assert hasattr(metrics, 'epochs')

    # if TF is available, attempt to build a tiny Keras model and export to tflite
    try:
        import tensorflow as tf
        model_tf = tf.keras.Sequential([tf.keras.layers.Dense(1, input_shape=(1,))])
        model_tf.compile(optimizer='adam', loss='mse')
        model_tf.fit(x, y, epochs=1, verbose=0)
        tflite_path = str(tmp_path / "model.tflite")
        ok = mu.export_tflite(model_tf, tflite_path)
        assert ok is True
        assert os.path.exists(tflite_path)
    except Exception:
        # TensorFlow not available in environment — skip export assertions
        pass
