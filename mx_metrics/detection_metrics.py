import logging
import mxnet as mx
import numpy as np

from .. import ROOT_LOGGER_NAME, ROOT_LOGGER_LEVEL
logger = logging.getLogger('{}.{}'.format(ROOT_LOGGER_NAME, __name__))
logger.setLevel(ROOT_LOGGER_LEVEL)


class BoxMAEMX(mx.metric.EvalMetric):
    """
    Use it only as epoch end metric
    """
    def __init__(self, name_prefix='', output_names=None, label_names=None):

        self.name = '{}_{}'.format(name_prefix, 'box_mae')

        self.preds = []
        self.trues = []

        self.num_inst = 0.
        self.num_images = 0.
        self.sum_metric = 0.

        super(BoxMAEMX, self).__init__(self.name, output_names=output_names, label_names=label_names)

    def update(self, labels, preds, **kwargs):
        """
        :param labels: [(batch_size, num_anchors, 4)] gt boxes
        :param preds: [(batch_size, num_anchors, 4)] predicted boxes
        :param kwargs: kwargs['mask'] - (batch_size, num_anchors) mask, whether is a gt box, if None, calculate on all
        """
        mask = kwargs.get('mask', mx.nd.ones_like(labels[0])).asnumpy()
        gt_boxes = labels[0].asnumpy()
        pred_boxes = preds[0].asnumpy()

        self.sum_metric += np.abs((gt_boxes - pred_boxes) * mask).sum()
        self.num_inst += np.sum(mask)  # mean per number of gt boxes
        self.num_images += gt_boxes.shape[0]

    def reset(self):
        if self.num_images != 0:
            logger.info('avg number of boxes per image: {}'.format(self.num_inst / self.num_images))
        self.num_images = 0
        super(BoxMAEMX, self).reset()


class BoxAccuracy(mx.metric.EvalMetric):
    def __init__(self, name_prefix='', output_names=None, label_names=None):
        self.name = '{}_{}'.format(name_prefix, 'box_acc')

        self.num_inst = 0.
        self.sum_metric = 0.

        super(BoxAccuracy, self).__init__(self.name, output_names=output_names, label_names=label_names)

    def update(self, labels, preds, **kwargs):
        """
        :param labels: [(batch_size, num_anchors)] gt boxes
        :param preds: [(batch_size, num_anchors, num_classes + 1)] predicted boxes
        :param kwargs: kwargs['mask'] - (batch_size, num_anchors) mask, whether is a gt box, if None, calculate on all
        """
        mask = kwargs.get('mask', mx.nd.ones_like(labels[0])).asnumpy()
        gt_classes = labels[0].asnumpy().astype(int)
        pred_classes_amax = np.argmax(preds[0].asnumpy(), axis=-1)

        self.sum_metric += np.abs((gt_classes == pred_classes_amax) * mask).sum()
        self.num_inst += np.sum(mask)  # mean per number of gt boxes
