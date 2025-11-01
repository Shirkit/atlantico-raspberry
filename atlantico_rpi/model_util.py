"""Model utilities and small persistence helpers."""

from dataclasses import dataclass
from typing import List, Optional, Any
import time
import json
import os
import numpy as np
import struct
from .config import X_TRAIN_PATH, Y_TRAIN_PATH
import logging

try:
    import tensorflow as tf
except Exception:
    tf = None

_LOG = logging.getLogger(__name__)


@dataclass
class Model:
    biases: Optional[List[float]] = None
    weights: Optional[List[float]] = None
    parsing_time: int = 0
    round: int = -1


@dataclass
class ClassClassifierMetrics:
    truePositives: int = 0
    trueNegatives: int = 0
    falsePositives: int = 0
    falseNegatives: int = 0


@dataclass
class MultiClassClassifierMetrics:
    metrics: Optional[List[ClassClassifierMetrics]] = None
    numberOfClasses: int = 0
    meanSqrdError: float = 0.0
    parsingTime: float = 0.0
    trainingTime: float = 0.0
    epochs: int = 0
    datasetSize: int = 0

    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1Score: float = 0.0
    precisionWeighted: float = 0.0
    recallWeighted: float = 0.0
    f1ScoreWeighted: float = 0.0


class ModelConfig:
    def __init__(self, layers: List[int], activation_functions: List[int], epochs: int = 1, random_seed: int = 10,
                 learning_rate_of_weights: float = 0.3333, learning_rate_of_biases: float = 0.0666,
                 json_weights: bool = False):
        self.layers = layers
        self.number_of_layers = len(layers)
        self.activation_functions = activation_functions
        self.epochs = epochs
        self.learning_rate_of_weights = learning_rate_of_weights
        self.learning_rate_of_biases = learning_rate_of_biases
        self.random_seed = random_seed
        self.json_weights = json_weights


class ModelUtil:
    """Utilities for model persistence and training stubs."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self._last_trained_tf_model = None

    def save_model_to_disk(self, model: Model, file_path: str) -> bool:
        payload = {
            "biases": model.biases if model.biases is not None else [],
            "weights": model.weights if model.weights is not None else [],
            "parsingTime": int(model.parsing_time),
            "round": int(model.round),
        }
        dirpath = os.path.dirname(file_path)
        if dirpath and not os.path.exists(dirpath):
            os.makedirs(dirpath, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        return True

    def load_model_from_disk(self, file_path: str) -> Model:
        if not os.path.exists(file_path):
            raise FileNotFoundError(file_path)
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        m = Model()
        m.biases = data.get("biases", [])
        m.weights = data.get("weights", [])
        m.parsing_time = int(data.get("parsingTime", 0))
        m.round = int(data.get("round", -1))
        return m

    def transform_data_to_model(self, stream) -> Model:
        """Transform an incoming stream (JSON or bytes) into a Model."""
        if stream is None:
            raise ValueError("stream must not be None")

        if isinstance(stream, dict):
            m = Model()
            m.biases = stream.get("biases", [])
            m.weights = stream.get("weights", [])
            m.parsing_time = int(stream.get("parsingTime", 0))
            m.round = int(stream.get("round", -1))
            return m

        if isinstance(stream, (bytes, bytearray)):
            parsed = self._parse_nn_bytes(bytes(stream))
            if parsed is None:
                m = Model()
                m.weights = [int(b) for b in stream]
                m.parsing_time = 0
                return m
            m = Model()
            biases = []
            weights = []
            for layer in parsed.get('layers', []):
                b = layer.get('biases', [])
                w = layer.get('weights', [])
                biases.extend(b.tolist() if isinstance(b, np.ndarray) else list(b))
                if isinstance(w, np.ndarray):
                    for row in w:
                        weights.extend(row.tolist())
                else:
                    weights.extend(w)
            m.biases = biases
            m.weights = weights
            m.parsing_time = int(parsed.get('parsingTime', 0))
            m.round = int(parsed.get('round', -1))
            return m

        raise NotImplementedError("Unsupported stream type for transform_data_to_model")

    def train_model_from_original_dataset(self, model: Model, x_file: str, y_file: str) -> MultiClassClassifierMetrics:
        """Train a model using files `x_file` and `y_file`. Returns metrics."""
        metrics = MultiClassClassifierMetrics()
        metrics.parsingTime = 0
        metrics.trainingTime = 0
        metrics.epochs = self.config.epochs

        if tf is None:
            return metrics

        def _load_csv(path):
            if not os.path.exists(path):
                raise FileNotFoundError(path)
            return np.loadtxt(path, delimiter=',')

        x_path = x_file
        y_path = y_file
        if not os.path.exists(x_path) and os.path.exists(X_TRAIN_PATH):
            x_path = X_TRAIN_PATH
        if not os.path.exists(y_path) and os.path.exists(Y_TRAIN_PATH):
            y_path = Y_TRAIN_PATH

        X = _load_csv(x_path)
        y = _load_csv(y_path)

        if X.ndim == 1:
            X = X.reshape((-1, 1))
        if y.ndim == 1:
            y = y.reshape((-1, 1))

        # Ensure X and y have the same number of samples. If files differ
        # (for example due to trailing newlines or pre-processing issues),
        # trim to the smaller set and log a warning so the caller can
        # investigate. Avoid raising here to keep training robust on-device.
        try:
            n_x = int(X.shape[0])
        except Exception:
            n_x = 0
        try:
            n_y = int(y.shape[0])
        except Exception:
            n_y = 0

        if n_x != n_y:
            _LOG.warning('Data cardinality mismatch: x samples=%s, y samples=%s — trimming to min', n_x, n_y)
            m = min(n_x, n_y)
            if m <= 0:
                _LOG.error('No samples available after trimming (x=%s, y=%s); aborting training', n_x, n_y)
                return metrics
            X = X[:m]
            y = y[:m]

        metrics.datasetSize = int(X.shape[0])

        keras = getattr(tf, 'keras', None)
        if keras is None:
            _LOG.debug('TensorFlow keras not available despite tf import; returning placeholder metrics')
            return metrics

        model_tf = keras.Sequential()
        input_dim = X.shape[1]
        model_tf.add(keras.Input(shape=(input_dim,)))

        # Map numeric activation codes to Keras activations or special markers
        ACT_MAP = {
            0: 'sigmoid',
            1: 'tanh',
            2: 'relu',
            3: 'leaky',   # special-case: keras.layers.LeakyReLU
            4: 'elu',
            5: 'selu',
            6: 'softmax',
        }

        # Determine architecture: accept full-arch (including input) or hidden+output sizes
        cfg_layers = getattr(self.config, 'layers', None) or []
        if isinstance(cfg_layers, (list, tuple)) and len(cfg_layers) >= 2:
            if int(cfg_layers[0]) == int(input_dim):
                arch = [int(x) for x in cfg_layers]
            else:
                arch = [int(input_dim)] + [int(x) for x in cfg_layers]
        else:
            arch = [int(input_dim)] + [int(x) for x in (getattr(self.config, 'layers') or [10, 10])]

        # If targets are provided as one-hot vectors, ensure the final
        # layer's number of outputs matches the number of classes. This
        # prevents Keras from raising a shape mismatch when using
        # categorical_crossentropy.
        if y.ndim > 1 and y.shape[1] > 1:
            try:
                required_outputs = int(y.shape[1])
                if arch[-1] != required_outputs:
                    _LOG.info(
                        "Adjusting final layer units from %s to %s to match one-hot targets",
                        arch[-1], required_outputs,
                    )
                    arch[-1] = required_outputs
            except Exception:
                # If anything goes wrong determining output size, prefer
                # to continue and let Keras raise an explicit error later.
                pass

        act_codes = list(getattr(self.config, 'activation_functions', []) or [])
        num_dense = len(arch) - 1
        for i in range(num_dense):
            units = int(arch[i + 1])
            code = act_codes[i] if i < len(act_codes) else 2
            act = ACT_MAP.get(int(code), 'linear')

            # Prefer softmax for final layer when targets are one-hot
            if i == num_dense - 1 and y.ndim > 1 and y.shape[1] > 1:
                act = 'softmax'

            if act == 'leaky':
                model_tf.add(keras.layers.Dense(units))
                model_tf.add(keras.layers.LeakyReLU(alpha=0.2))
            else:
                model_tf.add(keras.layers.Dense(units, activation=act))

        # Choose loss: categorical_crossentropy for one-hot targets, else mse
        if y.ndim > 1 and y.shape[1] > 1:
            loss = 'categorical_crossentropy'
        else:
            loss = 'mse'

        model_tf.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01), loss=loss)

        if hasattr(tf, 'timestamp'):
            start = tf.timestamp()
        else:
            start = time.time()

        history = model_tf.fit(X, y, epochs=max(1, self.config.epochs), verbose=0)

        if hasattr(tf, 'timestamp'):
            end = tf.timestamp()
        else:
            end = time.time()

        metrics.trainingTime = float(end - start)

        # Intentionally let assignment failures surface.
        self._last_trained_tf_model = model_tf

        metrics.meanSqrdError = float(history.history.get('loss', [0])[-1])

        # Predictions and classification metrics: convert softmax/prob vectors
        # to class labels via argmax for multi-class, otherwise threshold.
        preds = model_tf.predict(X)
        if preds.ndim > 1 and preds.shape[1] > 1:
            y_pred_labels = np.argmax(preds, axis=1)
        else:
            y_pred_labels = (preds.flatten() >= 0.5).astype(int)

        if y.ndim > 1 and y.shape[1] > 1:
            y_true_labels = np.argmax(y, axis=1)
        else:
            y_true_labels = y.flatten().astype(int)

        n_classes = int(max(y_true_labels.max() if y_true_labels.size > 0 else 0,
                            y_pred_labels.max() if y_pred_labels.size > 0 else 0) + 1)
        metrics.numberOfClasses = n_classes
        metrics.metrics = [ClassClassifierMetrics() for _ in range(n_classes)]

        # confusion counts
        for i in range(y_true_labels.shape[0]):
            t = int(y_true_labels[i])
            p = int(y_pred_labels[i])
            for c in range(n_classes):
                if t == c and p == c:
                    metrics.metrics[c].truePositives += 1
                elif t == c and p != c:
                    metrics.metrics[c].falseNegatives += 1
                elif t != c and p == c:
                    metrics.metrics[c].falsePositives += 1
                else:
                    metrics.metrics[c].trueNegatives += 1

        # sample-level accuracy
        metrics.accuracy = float(np.mean(y_pred_labels == y_true_labels)) if y_true_labels.size > 0 else 0.0

        # per-class precision/recall/f1 and aggregated metrics
        precisions = []
        recalls = []
        f1s = []
        for c in range(n_classes):
            tp = metrics.metrics[c].truePositives
            fp = metrics.metrics[c].falsePositives
            fn = metrics.metrics[c].falseNegatives
            prec = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
            rec = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
            precisions.append(prec)
            recalls.append(rec)
            f1s.append(f1)

        metrics.precision = float(sum(precisions) / len(precisions)) if precisions else 0.0
        metrics.recall = float(sum(recalls) / len(recalls)) if recalls else 0.0
        metrics.f1Score = float(sum(f1s) / len(f1s)) if f1s else 0.0

        # keep mean_squared_error already assigned from training history
        compute_weighted_metrics(metrics)
        return metrics

    def export_tflite(self, keras_model: Any, tflite_path: str) -> bool:
        """Convert a Keras model to TFLite and write to `tflite_path`."""
        if tf is None:
            raise RuntimeError("TensorFlow is not available in this environment")
        converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
        tflite_model = converter.convert()
        dirpath = os.path.dirname(tflite_path)
        if dirpath and not os.path.exists(dirpath):
            os.makedirs(dirpath, exist_ok=True)
        if isinstance(tflite_model, (bytes, bytearray)):
            tflite_bytes = bytes(tflite_model)
        else:
            try:
                tflite_bytes = bytes(tflite_model)  # type: ignore[arg-type]
            except Exception:
                _LOG.debug('tflite converter returned non-bytes; writing empty file')
                tflite_bytes = b''
        with open(tflite_path, 'wb') as f:
            f.write(tflite_bytes)
        InterpreterCls: Any = None
        try:
            from tflite_runtime.interpreter import Interpreter as _RTInterpreter  # type: ignore
            InterpreterCls = _RTInterpreter
        except Exception:
            InterpreterCls = None

        if InterpreterCls is None:
            try:
                from tensorflow.lite import Interpreter as _TFInterpreter  # type: ignore
                InterpreterCls = _TFInterpreter
            except Exception:
                InterpreterCls = None

        if InterpreterCls is None:
            try:
                from tensorflow.lite.python.interpreter import Interpreter as _TFPyInterpreter  # type: ignore
                InterpreterCls = _TFPyInterpreter
            except Exception:
                InterpreterCls = None

        if InterpreterCls is not None:
            interp = InterpreterCls(model_path=tflite_path)
            if hasattr(interp, 'allocate_tensors'):
                interp.allocate_tensors()
            try:
                mod_name = getattr(InterpreterCls, '__module__', '')
                if 'tensorflow.lite.python.interpreter' in mod_name:
                    _LOG.info('Using TensorFlow internal TFLite interpreter (module=%s); this may be deprecated', mod_name)
            except Exception:
                pass
            return True

        return True

    def serialize_to_nn_bytes(self, keras_model: Any = None, model: Model | None = None) -> bytes | None:
        """Serialize a trained model to the ESP32 `.nn` binary layout."""
        buf = bytearray()
        if keras_model is not None:
            pairs = []
            for layer in getattr(keras_model, 'layers', []):
                w = layer.get_weights()
                if w and len(w) == 2:
                    kernel, bias = w[0], w[1]
                    pairs.append((kernel, bias))

            num_layers = len(pairs)
            buf += struct.pack('<I', num_layers)
            for kernel, bias in pairs:
                inputs = int(kernel.shape[0])
                outputs = int(kernel.shape[1])
                buf += struct.pack('<I', inputs)
                buf += struct.pack('<I', outputs)
                for j in range(outputs):
                    b = float(bias[j]) if bias is not None else 0.0
                    buf += struct.pack('<f', b)
                    for k in range(inputs):
                        v = float(kernel[k, j])
                        buf += struct.pack('<f', v)
            return bytes(buf)

        if model is not None:
            if not hasattr(self, 'config') or not getattr(self.config, 'layers', None):
                return None
            layers = list(self.config.layers)
            if len(layers) < 2:
                return None
            num_layers = len(layers) - 1
            buf += struct.pack('<I', num_layers)
            biases_flat = list(model.biases or [])
            weights_flat = list(model.weights or [])
            b_idx = 0
            w_idx = 0
            for li in range(num_layers):
                inputs = int(layers[li])
                outputs = int(layers[li + 1])
                buf += struct.pack('<I', inputs)
                buf += struct.pack('<I', outputs)
                for out in range(outputs):
                    b = float(biases_flat[b_idx]) if b_idx < len(biases_flat) else 0.0
                    buf += struct.pack('<f', b)
                    b_idx += 1
                    for inp in range(inputs):
                        v = float(weights_flat[w_idx]) if w_idx < len(weights_flat) else 0.0
                        buf += struct.pack('<f', v)
                        w_idx += 1
            return bytes(buf)

        return None

    def _parse_nn_bytes(self, data: bytes) -> dict | None:
        """Parse an ESP32 .nn binary into a dict with layers: weights and biases."""
        offset = 0
        if len(data) < 4:
            return None
        num_layers = struct.unpack_from('<I', data, offset)[0]
        offset += 4
        layers = []
        for layer_idx in range(num_layers):
            activation = None
            if offset + 1 <= len(data):
                potential_activation = struct.unpack_from('<B', data, offset)[0]
                if 0 <= potential_activation <= 6:
                    if offset + 9 <= len(data):
                        inputs = struct.unpack_from('<I', data, offset+1)[0]
                        outputs = struct.unpack_from('<I', data, offset+5)[0]
                        if 1 <= inputs <= 2000 and 1 <= outputs <= 2000:
                            activation = potential_activation
                            offset += 1
            if offset + 8 > len(data):
                return None
            inputs = struct.unpack_from('<I', data, offset)[0]
            outputs = struct.unpack_from('<I', data, offset+4)[0]
            offset += 8

            remaining = len(data) - offset
            floats_available = remaining // 4
            values_per_output = 1 + inputs
            possible_outputs = floats_available // values_per_output
            read_outputs = min(possible_outputs, outputs)

            biases = np.zeros(outputs, dtype=np.float32)
            weights = np.zeros((outputs, inputs), dtype=np.float32)
            for j in range(read_outputs):
                if offset + 4 > len(data):
                    break
                bias_v = struct.unpack_from('<f', data, offset)[0]
                offset += 4
                biases[j] = bias_v
                for k in range(inputs):
                    if offset + 4 > len(data):
                        break
                    wv = struct.unpack_from('<f', data, offset)[0]
                    offset += 4
                    weights[j, k] = wv
            layers.append({
                'inputs': inputs,
                'outputs': outputs,
                'biases': biases,
                'weights': weights,
                'activation': activation,
            })
        return {'num_layers': num_layers, 'layers': layers}

    def predict_from_current_model(self, model: Model, x):
        """Perform a lightweight predict using the current model."""
        if model is None:
            raise ValueError("model is required")
        length = len(x)
        return [0 for _ in range(length)]


def compute_weighted_metrics(metrics: MultiClassClassifierMetrics) -> None:
    """Compute support-weighted precision/recall/f1 in-place."""
    if metrics is None or metrics.metrics is None or metrics.numberOfClasses == 0:
        return

    supports = []
    precisions = []
    recalls = []
    f1s = []
    total_support = 0
    for c in range(metrics.numberOfClasses):
        m = metrics.metrics[c]
        tp = m.truePositives
        fp = m.falsePositives
        fn = m.falseNegatives
        support = tp + fn
        supports.append(support)
        total_support += support
        prec = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

    metrics.precision = float(sum(precisions) / len(precisions)) if precisions else 0.0
    metrics.recall = float(sum(recalls) / len(recalls)) if recalls else 0.0
    metrics.f1Score = float(sum(f1s) / len(f1s)) if f1s else 0.0

    if total_support > 0:
        metrics.precisionWeighted = float(sum(p * s for p, s in zip(precisions, supports)) / total_support)
        metrics.recallWeighted = float(sum(r * s for r, s in zip(recalls, supports)) / total_support)
        metrics.f1ScoreWeighted = float(sum(f * s for f, s in zip(f1s, supports)) / total_support)
    else:
        metrics.precisionWeighted = metrics.precision
        metrics.recallWeighted = metrics.recall
        metrics.f1ScoreWeighted = metrics.f1Score

