import cv2
import numpy as np

from .base_preprocessor import MIMOPreprocessor


class BoxLabelGetter(MIMOPreprocessor):
    def __init__(self, func, data_names=('boxes', 'labels'), input_names=('label',), *args, **kwargs):
        super(BoxLabelGetter, self).__init__(data_names=data_names, input_names=input_names, *args, **kwargs)
        self._getter = func

    def process(self, **kwargs):
        boxes, labels = self._getter(kwargs[self.provide_input[0]])
        return {self.provide_output[0]: boxes, self.provide_output[1]: labels}


class BoxLabelPreprocessor(MIMOPreprocessor):
    def __init__(self, func, input_names=('boxes', 'labels'), *args, **kwargs):
        super(BoxLabelPreprocessor, self).__init__(input_names=input_names, *args, **kwargs)
        self._transformer = func

    def process(self, **kwargs):
        bundle = self._transformer(**kwargs)
        return {name: data for name, data in zip(self.provide_output, bundle)}


class BoxImagePreprocessor(MIMOPreprocessor):
    def __init__(self, func, input_names=('image', 'boxes'), data_names=('image', 'boxes'), *args, **kwargs):
        super(BoxImagePreprocessor, self).__init__(input_names=input_names, data_names=data_names, *args, **kwargs)
        self._transformer = func

    def process(self, **kwargs):
        bundle = self._transformer(**kwargs)
        return {name: data for name, data in zip(self.provide_output, bundle)}


class BoxLabelBatchify(MIMOPreprocessor):

    def __init__(self, max_boxes=20, input_names=('boxes', 'labels'), data_names=('boxes', 'labels'), *args, **kwargs):
        super(BoxLabelBatchify, self).__init__(input_names=input_names, data_names=data_names,
                                               data_shapes=[(max_boxes, 4), (max_boxes, 1)], *args, **kwargs)
        self._max_boxes = max_boxes

    def process(self, **kwargs):
        boxes = kwargs[self.provide_input[0]]
        labels = kwargs[self.provide_input[1]]

        labels_batched = - np.ones((self._max_boxes, 1))
        boxes_batched = np.zeros((self._max_boxes, 4))

        labels_batched[:min(labels.shape[0], self._max_boxes), :] = labels[:min(labels.shape[0], self._max_boxes)]
        boxes_batched[:min(boxes.shape[0], self._max_boxes), :] = boxes[:min(boxes.shape[0], self._max_boxes)]

        return {name: pack for name, pack in zip(self.provide_output, [boxes_batched, labels_batched])}


class CropRGBImage(MIMOPreprocessor):
    def __init__(self, data_names=('image',), input_names=('image', 'crop'), *args, **kwargs):
        super(CropRGBImage, self).__init__(data_names=data_names, input_names=input_names, *args, **kwargs)

    def process(self, **kwargs):
        # crop is x, y, x, y, image is hwc
        image, crop = kwargs[self.provide_input[0]], kwargs[self.provide_input[1]]
        cropped = image[slice(*crop[1::2]), slice(*crop[0::2])]
        return {self.provide_output[0]: cropped}


class BoxImageFlip(MIMOPreprocessor):
    def __init__(self, lr_flip_prob=0.5, ud_flip_prob=0.5, input_names=('image', 'boxes'),
                 data_names=('image', 'boxes'), *args, **kwargs):
        super(BoxImageFlip, self).__init__(input_names=input_names, data_names=data_names, *args, **kwargs)

        self._lr_prob = lr_flip_prob
        self._ud_prob = ud_flip_prob

    def process(self, **kwargs):
        # boxes is x, y, x, y, image is hwc
        image, boxes = kwargs[self.provide_input[0]], kwargs[self.provide_input[1]]
        to_crop_ud = np.random.random() < self._ud_prob
        to_crop_lr = np.random.random() < self._lr_prob

        if to_crop_ud:
            boxes[:, 1::2] = 1 - boxes[:, 1::2]
            to_swap1, to_swap2 = np.copy(boxes[:, 1]), np.copy(boxes[:, 3])
            boxes[:, 1], boxes[:, 3] = to_swap2, to_swap1
            image = image[::-1, :, :]

        if to_crop_lr:
            boxes[:, 0::2] = 1 - boxes[:, 0::2]
            to_swap1, to_swap2 = np.copy(boxes[:, 0]), np.copy(boxes[:, 2])
            boxes[:, 0], boxes[:, 2] = to_swap2, to_swap1
            image = image[:, ::-1, :]

        return {self.provide_output[0]: image, self.provide_output[1]: boxes}