import logging
import mxnet as mx
import numpy as np
from sklearn.metrics import accuracy_score, log_loss


from .. import ROOT_LOGGER_NAME, ROOT_LOGGER_LEVEL
logger = logging.getLogger('{}.{}'.format(ROOT_LOGGER_NAME, __name__))
logger.setLevel(ROOT_LOGGER_LEVEL)


class SklearnReportMX(mx.metric.EvalMetric):
    """
    Use it only as epoch end metric
    """
    def __init__(self, sklearn_metric_func, labels_count, name_prefix='',
                 output_names=None, label_names=None):

        self.labels_count = labels_count
        self._metric_func = sklearn_metric_func
        self.name = '{}_{}'.format(name_prefix, sklearn_metric_func.__name__)
        self.preds = []
        self.trues = []

        self.num_inst = 0
        self.sum_metric = 0.

        super(SklearnReportMX, self).__init__(self.name, output_names=output_names, label_names=label_names)

    def update(self, labels, preds):
        labels = labels[0].asnumpy()
        preds = preds[0].asnumpy()

        self.trues.append(labels)
        self.preds.append(preds)

        self.num_inst += len(labels)
        print('kek')

    def reset(self):
        try:
            cum_true = np.concatenate(self.trues)
            cum_pred = np.concatenate(self.preds)

            logger.info(self._metric_func(
                y_true=np.argmax(cum_true, axis=1),
                y_pred=np.argmax(cum_pred, axis=1),
                labels=range(self.labels_count))
            )
        except ValueError:
            logger.debug('metric {} reset error'.format(self._metric_func.__name__))

        self.trues = []
        self.preds = []

        super(SklearnReportMX, self).reset()


def get_accuracy(y_label, y_pred):
    ids_label = np.argmax(y_label, axis=1)
    ids_pred = np.argmax(y_pred, axis=1)
    logger.debug('passed predictions and trues: {}, {}'.format(ids_pred, ids_label))
    return accuracy_score(y_pred=ids_pred, y_true=ids_label)


def make_log_loss(labels):
    def get_log_loss(y_label, y_pred):
        ids_label = np.argmax(y_label, axis=1)
        logger.debug('passed predictions and trues: {}, {}'.format(y_pred, ids_label))
        return log_loss(y_pred=y_pred, y_true=ids_label, labels=labels)
    return get_log_loss
