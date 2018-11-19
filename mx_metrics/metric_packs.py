import mxnet as mx
import logging
from sklearn.metrics import classification_report, confusion_matrix

from .sigmoid_metrics import get_per_class_metric_per_threshold
from .softmax_metrics import SklearnReportMX, get_accuracy, make_log_loss

from .. import ROOT_LOGGER_NAME, ROOT_LOGGER_LEVEL
logger = logging.getLogger('{}.{}'.format(ROOT_LOGGER_NAME, __name__))
logger.setLevel(ROOT_LOGGER_LEVEL)


def get_multi_class_metric_pack(sklearn_binary_metric, output_names, label_names, classes,
                                thresholds=(0.1, 0.3, 0.5, 0.7, 0.9)):
    metrics = []

    metric_func = sklearn_binary_metric
    metric_func_name = metric_func.__name__

    metric_funcs = [[get_per_class_metric_per_threshold(
        class_num=cl, metric_func=metric_func, threshold=th,
        metric_func_params={'average': 'micro' if cl is None else 'binary'}
    ) for th in thresholds] for cl in classes]  # it is a list of lists

    metrics += [mx.metric.CustomMetric(
        feval=f, name='\n{}_cl{}_th{}'.format(metric_func_name, cl, th),
        output_names=output_names,
        label_names=label_names)
        for cl, fc in zip(classes, metric_funcs) for f, th in zip(fc, thresholds)]

    return metrics


def get_single_class_metric_pack(output_names, label_names, classes, name=''):
    metrics = []

    metrics += [mx.metric.CustomMetric(
        feval=get_accuracy, name='\n{} accuracy'.format(name),
        output_names=output_names,
        label_names=label_names)]

    metrics += [mx.metric.CustomMetric(
        feval=make_log_loss(classes), name='\n{} log_loss'.format(name),
        output_names=output_names,
        label_names=label_names)]

    sklearn_metric_func = [confusion_matrix, classification_report]

    metrics += [
        SklearnReportMX(f, len(classes), output_names=output_names, label_names=label_names, name_prefix=name)
        for f in sklearn_metric_func
    ]

    return metrics
