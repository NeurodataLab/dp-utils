import logging

from .. import ROOT_LOGGER_NAME, ROOT_LOGGER_LEVEL
logger = logging.getLogger('{}.{}'.format(ROOT_LOGGER_NAME, __name__))
logger.setLevel(ROOT_LOGGER_LEVEL)


def get_per_class_metric_per_threshold(metric_func, class_num=None, freeze=(True, True),
                                       threshold=0.5, metric_func_params=None):
    metric_func_params = metric_func_params or {}

    def wrapped(y_label, y_pred):
        if freeze[0]:
            y_label = (y_label > threshold).astype(int)

        if freeze[1]:
            y_pred = (y_pred > threshold).astype(int)

        if class_num is None:
            return metric_func(y_true=y_label, y_pred=y_pred, **metric_func_params)
        else:
            return metric_func(y_true=y_label[:, class_num], y_pred=y_pred[:, class_num], **metric_func_params)

    return wrapped
