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


class BoxClassPRBase(mx.metric.EvalMetric):
    def __init__(self, class_index=1, class_threshold=0.5, output_names=None, label_names=None):

        self.num_inst = 0.
        self.sum_metric = 0.

        self._tp = 0.
        self._fp = 0.
        self._fn = 0.

        self._threshold = class_threshold
        self._class_id = class_index

        super(BoxClassPRBase, self).__init__(self.name, output_names=output_names, label_names=label_names)

    def update(self, labels, preds, **kwargs):
        """
        :param labels: [(batch_size, num_anchors)] gt boxes
        :param preds: [(batch_size, num_anchors, num_classes + 1)] predicted boxes
        :param kwargs: kwargs['mask'] - (batch_size, num_anchors) mask, whether is a gt box, if None, calculate on all
        """
        mask = kwargs.get('mask', mx.nd.ones_like(labels[0])).asnumpy().astype(bool)
        pred_probs = mx.nd.softmax(preds[0], axis=-1)
        gt_classes = labels[0].asnumpy().astype(int).flatten()[mask.flatten()]

        class_mask = pred_probs[:, :, self._class_id].asnumpy().flatten()[mask.flatten()] > self._threshold

        self._tp += (gt_classes[class_mask] == self._class_id).sum()
        self._fp += (gt_classes[class_mask] != self._class_id).sum()
        self._fn += (gt_classes[np.logical_not(class_mask)] == self._class_id).sum()

        self.num_inst += class_mask.sum()

    def reset(self):
        super(BoxClassPRBase, self).reset()
        self._fp, self._tp, self._fn = 0., 0., 0.


class BoxClassPrecision(BoxClassPRBase):
    def __init__(self, name_prefix='', *args, **kwargs):
        self.name = '{}_{}'.format(name_prefix, 'precision')

        super(BoxClassPrecision, self).__init__(*args, **kwargs)

    def get(self):
        return self.name, self._tp / (self._fp + self._tp + 1e-8)


class BoxClassRecall(BoxClassPRBase):
    def __init__(self, name_prefix='', *args, **kwargs):
        self.name = '{}_{}'.format(name_prefix, 'recall')

        super(BoxClassRecall, self).__init__(*args, **kwargs)

    def get(self):
        return self.name, self._tp / (self._fn + self._tp + 1e-8)
