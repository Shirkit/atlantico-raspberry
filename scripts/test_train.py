#!/usr/bin/env python3
"""Standalone test runner that invokes train_model_from_original_dataset
and writes JSON metrics to run/logs.
"""
import argparse
import json
import os
import sys
import time
import traceback

# Ensure repo root is on sys.path so `import atlantico_rpi` works when this
# script is executed from the scripts/ directory.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from atlantico_rpi.config import X_TRAIN_PATH, Y_TRAIN_PATH
from atlantico_rpi.model_util import ModelConfig, ModelUtil, Model


def metrics_to_dict(metrics):
    d = {k: getattr(metrics, k) for k in ['number_of_classes', 'mean_squared_error', 'parsing_time', 'training_time', 'epochs', 'accuracy', 'precision', 'recall', 'f1Score', 'meanSqrdError', 'precision_weighted', 'recall_weighted', 'f1Score_weighted']}
    mm = []
    if getattr(metrics, 'metrics', None):
        for c in metrics.metrics:
            mm.append({'true_positives': c.true_positives, 'true_negatives': c.true_negatives, 'false_positives': c.false_positives, 'false_negatives': c.false_negatives})
    d['per_class'] = mm
    return d


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--layers', type=str, default='16,16', help='comma separated layer sizes')
    p.add_argument('--outdir', type=str, default=os.path.join('run', 'logs'))
    args = p.parse_args()

    layers = [int(x) for x in args.layers.split(',') if x.strip()]
    mc = ModelConfig(layers=layers, activation_functions=[0] * len(layers), epochs=args.epochs)
    util = ModelUtil(mc)
    m = Model()

    try:
        print('Using X_TRAIN_PATH=', X_TRAIN_PATH, 'Y_TRAIN_PATH=', Y_TRAIN_PATH)
        metrics = util.train_model_from_original_dataset(m, X_TRAIN_PATH, Y_TRAIN_PATH)
        out = metrics_to_dict(metrics)
        out['timestamp'] = time.time()

        os.makedirs(args.outdir, exist_ok=True)
        fname = os.path.join(args.outdir, f'test_train_{int(time.time())}.json')
        with open(fname, 'w') as f:
            json.dump(out, f, indent=2)
        print(json.dumps(out, indent=2))
        print('Wrote metrics to', fname)
    except Exception:
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()
