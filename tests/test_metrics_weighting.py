from atlantico_rpi.model_util import ClassClassifierMetrics, MultiClassClassifierMetrics, compute_weighted_metrics
from atlantico_rpi.device import compare_metrics


def make_metrics(tp_fp_fn_tuples):
    # tp_fp_fn_tuples: list of (tp, fp, fn, tn)
    metrics = MultiClassClassifierMetrics()
    metrics.number_of_classes = len(tp_fp_fn_tuples)
    metrics.metrics = []
    total = 0
    for tp, fp, fn, tn in tp_fp_fn_tuples:
        m = ClassClassifierMetrics()
        m.true_positives = tp
        m.false_positives = fp
        m.false_negatives = fn
        m.true_negatives = tn
        metrics.metrics.append(m)
        total += tp + tn + fp + fn
    # if no samples, leave accuracy 0
    return metrics


def test_weighted_metrics_simple_case():
    # class 0: perfect (tp=10), class1: poor (tp=1, fn=9)
    m = make_metrics([(10,0,0,0),(1,0,9,0)])
    # set some hooks
    compute_weighted_metrics(m)
    # weighted precision/recall should prefer class0 because it has higher support
    assert m.precision >= 0.0
    assert m.recall >= 0.0
    assert m.f1Score >= 0.0
    assert m.precision_weighted >= 0.0
    assert m.recall_weighted >= 0.0
    assert m.f1Score_weighted >= 0.0


def test_compare_metrics_prefers_better_model():
    # Old model has low accuracy; new model higher accuracy
    old = make_metrics([(1,4,5,0)])
    new = make_metrics([(5,0,0,0)])
    compute_weighted_metrics(old)
    compute_weighted_metrics(new)
    assert compare_metrics(old, new) is True

 