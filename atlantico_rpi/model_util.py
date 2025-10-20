"""Model utilities and small persistence helpers."""

from dataclasses import dataclass
from typing import List, Optional
from typing import Any
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

# module logger
_LOG = logging.getLogger(__name__)


@dataclass
class Model:
    biases: Optional[List[float]] = None
    weights: Optional[List[float]] = None
    parsing_time: int = 0
    round: int = -1


@dataclass
class ClassClassifierMetrics:
    true_positives: int = 0
    true_negatives: int = 0
    false_positives: int = 0
    false_negatives: int = 0


@dataclass
class MultiClassClassifierMetrics:
    metrics: Optional[List[ClassClassifierMetrics]] = None
    number_of_classes: int = 0
    mean_squared_error: float = 0.0
    parsing_time: float = 0.0
    training_time: float = 0.0
    epochs: int = 0

    # Derived summary metrics (populated after training)
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1Score: float = 0.0
    # compatibility alias used elsewhere
    meanSqrdError: float = 0.0
    # Weighted versions (support-weighted)
    precision_weighted: float = 0.0
    recall_weighted: float = 0.0
    f1Score_weighted: float = 0.0
    # Simple averaged metric (mean of accuracy, precision, recall, f1 macro)


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

    def save_model_to_disk(self, model: Model, file_path: str) -> bool:
        try:
            payload = {
                "biases": model.biases if model.biases is not None else [],
                "weights": model.weights if model.weights is not None else [],
                "parsing_time": int(model.parsing_time),
                "round": int(model.round),
            }
            dirpath = os.path.dirname(file_path)
            if dirpath and not os.path.exists(dirpath):
                os.makedirs(dirpath, exist_ok=True)
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(payload, f)
            return True
        except Exception:
            return False

    def load_model_from_disk(self, file_path: str) -> Model:
        if not os.path.exists(file_path):
            raise FileNotFoundError(file_path)
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        m = Model()
        m.biases = data.get("biases", [])
        m.weights = data.get("weights", [])
        m.parsing_time = int(data.get("parsing_time", 0))
        m.round = int(data.get("round", -1))
        return m

    def transform_data_to_model(self, stream) -> Model:
        """Transform an incoming stream (e.g., JSON or bytes) into a Model.

        This is a light-weight stub: real implementations should parse the
        payload (JSON or binary) and populate a Model instance. We accept
        either a dict-like object or raw bytes.
        """
        if stream is None:
            raise ValueError("stream must not be None")

        # if dict-like, assume it is already the model payload
        if isinstance(stream, dict):
            m = Model()
            m.biases = stream.get("biases", [])
            m.weights = stream.get("weights", [])
            m.parsing_time = int(stream.get("parsing_time", 0))
            m.round = int(stream.get("round", -1))
            return m

        # if bytes, attempt to parse ESP32 .nn binary format
        if isinstance(stream, (bytes, bytearray)):
            parsed = self._parse_nn_bytes(bytes(stream))
            if parsed is None:
                # fallback: store raw bytes as ints
                m = Model()
                m.weights = [int(b) for b in stream]
                m.parsing_time = 0
                return m
            # produce Model with flattened weights/biases
            m = Model()
            biases = []
            weights = []
            for layer in parsed.get('layers', []):
                # append biases then weights in same order as ESP32 reader
                b = layer.get('biases', [])
                w = layer.get('weights', [])
                biases.extend(b.tolist() if isinstance(b, np.ndarray) else list(b))
                # flatten weights (weights is numpy array shape outputs x inputs)
                if isinstance(w, np.ndarray):
                    # iterate outputs then inputs
                    for row in w:
                        weights.extend(row.tolist())
                else:
                    weights.extend(w)
            m.biases = biases
            m.weights = weights
            m.parsing_time = int(parsed.get('parsing_time', 0))
            m.round = int(parsed.get('round', -1))
            return m

        raise NotImplementedError("Unsupported stream type for transform_data_to_model")

    from typing import Any

    def train_model_from_original_dataset(self, model: Model, x_file: str, y_file: str) -> MultiClassClassifierMetrics:
        """Train a model using files `x_file` and `y_file`.

        Minimal placeholder implementation: it does not train a neural net but
        returns an empty metrics object and increments parsing_time.
        """
        # If TensorFlow is not available, return a placeholder metrics object.
        metrics = MultiClassClassifierMetrics()
        metrics.parsing_time = 0
        metrics.training_time = 0
        metrics.epochs = self.config.epochs

        if tf is None:
            return metrics

        # simple CSV loader: expect numeric CSVs with rows=samples
        def _load_csv(path):
            if not os.path.exists(path):
                raise FileNotFoundError(path)
            return np.loadtxt(path, delimiter=',')

        # attempt to load CSV files; if they are missing or unreadable, return
        # the placeholder metrics so callers/tests that don't provide dataset
        # files still get a sensible result.
        # allow callers to pass short filenames; if they don't exist, fallback
        # to configured dataset paths in `atlantico_rpi/config.py`.
        x_path = x_file
        y_path = y_file
        if not os.path.exists(x_path) and os.path.exists(X_TRAIN_PATH):
            x_path = X_TRAIN_PATH
        if not os.path.exists(y_path) and os.path.exists(Y_TRAIN_PATH):
            y_path = Y_TRAIN_PATH

        try:
            X = _load_csv(x_path)
            y = _load_csv(y_path)
        except FileNotFoundError:
            # missing dataset files: return placeholder metrics
            return metrics

        # ensure shape (n_samples, n_features)
        if X.ndim == 1:
            X = X.reshape((-1, 1))
        if y.ndim == 1:
            y = y.reshape((-1, 1))

        # build a tiny sequential model based on config, using an explicit Input
        # layer to avoid Keras input_shape deprecation warnings.
        keras = getattr(tf, 'keras', None)
        if keras is None:
            _LOG.debug('TensorFlow keras not available despite tf import; returning placeholder metrics')
            return metrics

        model_tf = keras.Sequential()
        input_dim = X.shape[1]
        model_tf.add(keras.Input(shape=(input_dim,)))
        for units in self.config.layers:
            model_tf.add(keras.layers.Dense(units, activation='relu'))
        # final layer: match y dims
        model_tf.add(keras.layers.Dense(y.shape[1], activation='linear'))

        model_tf.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01), loss='mse')

        # prefer tf.timestamp when available for more accurate timing
        if hasattr(tf, 'timestamp'):
            try:
                start = tf.timestamp()
            except Exception:
                start = time.time()
        else:
            start = time.time()
        history = model_tf.fit(X, y, epochs=max(1, self.config.epochs), verbose=0)
        if hasattr(tf, 'timestamp'):
            try:
                end = tf.timestamp()
            except Exception:
                end = time.time()
        else:
            end = time.time()

        metrics.training_time = float(end - start)
        metrics.mean_squared_error = float(history.history.get('loss', [0])[-1])
        metrics.meanSqrdError = metrics.mean_squared_error

        # compute predictions and per-class confusion counts (TP/FP/TN/FN)
        try:
            preds = model_tf.predict(X)
            # ensure shapes
            if preds.ndim == 1:
                preds = preds.reshape((-1, 1))

            # y may be float; threshold at 0.5 for binary decisions
            y_bin = (y >= 0.5).astype(int)
            y_pred_bin = (preds >= 0.5).astype(int)

            n_classes = y_bin.shape[1]
            metrics.number_of_classes = n_classes
            metrics.metrics = [ClassClassifierMetrics() for _ in range(n_classes)]

            total_tp = total_tn = total_fp = total_fn = 0
            total_elements = 0
            # compute MSE across outputs as well
            mse = 0.0
            for i in range(y_bin.shape[0]):
                for c in range(n_classes):
                    yt = int(y_bin[i, c])
                    yp = int(y_pred_bin[i, c])
                    total_elements += 1
                    if yt == 1 and yp == 1:
                        metrics.metrics[c].true_positives += 1
                        total_tp += 1
                    elif yt == 1 and yp == 0:
                        metrics.metrics[c].false_negatives += 1
                        total_fn += 1
                    elif yt == 0 and yp == 1:
                        metrics.metrics[c].false_positives += 1
                        total_fp += 1
                    else:
                        metrics.metrics[c].true_negatives += 1
                        total_tn += 1
                # accumulate mse
                diff = y[i] - preds[i]
                mse += float((diff ** 2).mean())

            if total_elements > 0:
                metrics.accuracy = float((total_tp + total_tn) / total_elements)
            else:
                metrics.accuracy = 0.0

            # per-class precision/recall -> macro-average
            precisions = []
            recalls = []
            f1s = []
            for c in range(n_classes):
                tp = metrics.metrics[c].true_positives
                fp = metrics.metrics[c].false_positives
                fn = metrics.metrics[c].false_negatives
                prec = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
                rec = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
                f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
                precisions.append(prec)
                recalls.append(rec)
                f1s.append(f1)

            metrics.precision = float(sum(precisions) / len(precisions)) if precisions else 0.0
            metrics.recall = float(sum(recalls) / len(recalls)) if recalls else 0.0
            metrics.f1Score = float(sum(f1s) / len(f1s)) if f1s else 0.0

            # average mse per sample
            metrics.mean_squared_error = float(mse / max(1, y.shape[0]))
            metrics.meanSqrdError = metrics.mean_squared_error
            # compute weighted metrics and combined scores
            try:
                compute_weighted_metrics(metrics)
            except Exception:
                pass
        except Exception:
            # if any of the prediction/metric computation fails, keep placeholders
            pass
        return metrics

    def export_tflite(self, keras_model: Any, tflite_path: str) -> bool:
        """Convert a Keras model to TFLite and write to `tflite_path`."""
        if tf is None:
            raise RuntimeError("TensorFlow is not available in this environment")
        try:
            converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
            tflite_model = converter.convert()
            dirpath = os.path.dirname(tflite_path)
            if dirpath and not os.path.exists(dirpath):
                os.makedirs(dirpath, exist_ok=True)
            # ensure we have a bytes-like object to write
            # ensure we have a bytes-like object to write
            if isinstance(tflite_model, (bytes, bytearray)):
                tflite_bytes = bytes(tflite_model)
            else:
                try:
                    # some converters return array('b') or memoryview-like objects
                    tflite_bytes = bytes(tflite_model)  # type: ignore[arg-type]
                except Exception:
                    _LOG.debug('tflite converter returned non-bytes; writing empty file')
                    tflite_bytes = b''
            with open(tflite_path, 'wb') as f:
                f.write(tflite_bytes)
            # validate that we can load the produced tflite with an available
            # interpreter implementation. Prefer `tflite_runtime` when present,
            # fall back to TensorFlow's interpreter implementation.
            InterpreterCls: Any = None
            # Try tflite_runtime first
            try:
                from tflite_runtime.interpreter import Interpreter as _RTInterpreter  # type: ignore
                InterpreterCls = _RTInterpreter
            except Exception:
                InterpreterCls = None

            if InterpreterCls is None:
                # Try tensorflow.lite (may not expose Interpreter at top-level)
                try:
                    from tensorflow.lite import Interpreter as _TFInterpreter  # type: ignore
                    InterpreterCls = _TFInterpreter
                except Exception:
                    InterpreterCls = None

            if InterpreterCls is None:
                try:
                    # Last resort: internal path that is present in many TF wheels
                    from tensorflow.lite.python.interpreter import Interpreter as _TFPyInterpreter  # type: ignore
                    InterpreterCls = _TFPyInterpreter
                except Exception:
                    InterpreterCls = None

            if InterpreterCls is not None:
                try:
                    # load by filename to ensure runtime can parse
                    interp = InterpreterCls(model_path=tflite_path)
                    # allocate tensors (some interpreters require this step)
                    if hasattr(interp, 'allocate_tensors'):
                        interp.allocate_tensors()
                    # If we reached here and the interpreter was the TF internal
                    # python interpreter, it may be deprecated; log at INFO so the
                    # developer can decide to migrate to `tflite-runtime` or
                    # LiteRT in the future.
                    try:
                        mod_name = getattr(InterpreterCls, '__module__', '')
                        if 'tensorflow.lite.python.interpreter' in mod_name:
                            _LOG.info('Using TensorFlow internal TFLite interpreter (module=%s); this may be deprecated', mod_name)
                    except Exception:
                        pass
                    return True
                except Exception:
                    # If validation fails, still keep the file but indicate failure
                    return False

            # No interpreter available to validate, but the file was written.
            return True
        except Exception:
            return False

    def _parse_nn_bytes(self, data: bytes) -> dict | None:
        """Parse an ESP32 .nn binary into a dict with layers: weights (numpy) and biases (numpy).

        This is a defensive, heuristic parser ported/simplified from the server reader.
        Returns None on failure.
        """
        try:
            offset = 0
            if len(data) < 4:
                return None
            num_layers = struct.unpack_from('<I', data, offset)[0]
            offset += 4
            layers = []
            for layer_idx in range(num_layers):
                # try to detect optional activation byte
                activation = None
                if offset + 1 <= len(data):
                    potential_activation = struct.unpack_from('<B', data, offset)[0]
                    if 0 <= potential_activation <= 6:
                        # peek next 8 bytes to verify dims
                        if offset + 9 <= len(data):
                            inputs = struct.unpack_from('<I', data, offset+1)[0]
                            outputs = struct.unpack_from('<I', data, offset+5)[0]
                            if 1 <= inputs <= 2000 and 1 <= outputs <= 2000:
                                activation = potential_activation
                                offset += 1
                # read inputs and outputs
                if offset + 8 > len(data):
                    return None
                inputs = struct.unpack_from('<I', data, offset)[0]
                outputs = struct.unpack_from('<I', data, offset+4)[0]
                offset += 8

                # compute how many floats available
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
        except Exception:
            return None

    def predict_from_current_model(self, model: Model, x):
        """Perform a lightweight predict using the current model.

        Placeholder: returns zeros with the same length as input.
        """
        if model is None:
            raise ValueError("model is required")
        try:
            length = len(x)
            return [0 for _ in range(length)]
        except Exception:
            # if x is a single value
            return 0


def compute_weighted_metrics(metrics: MultiClassClassifierMetrics) -> None:
    """Compute support-weighted precision/recall/f1 and combined metrics in-place.

    Weighted metrics use per-class support (TP+FN) as weight. The function
    populates precision_weighted, recall_weighted, and f1Score_weighted.
    """
    if metrics is None or metrics.metrics is None or metrics.number_of_classes == 0:
        return

    supports = []
    precisions = []
    recalls = []
    f1s = []
    total_support = 0
    for c in range(metrics.number_of_classes):
        m = metrics.metrics[c]
        tp = m.true_positives
        fp = m.false_positives
        fn = m.false_negatives
        support = tp + fn
        supports.append(support)
        total_support += support
        prec = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

    # macro (simple) averages
    metrics.precision = float(sum(precisions) / len(precisions)) if precisions else 0.0
    metrics.recall = float(sum(recalls) / len(recalls)) if recalls else 0.0
    metrics.f1Score = float(sum(f1s) / len(f1s)) if f1s else 0.0

    # weighted by support
    if total_support > 0:
        metrics.precision_weighted = float(sum(p * s for p, s in zip(precisions, supports)) / total_support)
        metrics.recall_weighted = float(sum(r * s for r, s in zip(recalls, supports)) / total_support)
        metrics.f1Score_weighted = float(sum(f * s for f, s in zip(f1s, supports)) / total_support)
    else:
        metrics.precision_weighted = metrics.precision
        metrics.recall_weighted = metrics.recall
        metrics.f1Score_weighted = metrics.f1Score

    # Note: combined metrics removed per research request; only per-field and
    # support-weighted precision/recall/f1 are retained.

