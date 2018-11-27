import cv2
import logging

import numpy as np
import os

from .base_preprocessor import BasePreprocessor
from ...data_iterators import TRIAL_DATA_DIR
from ...image_transformers.resizing import loop_video_size_casting, back_and_fourth_video_size_casting, \
    make_random_beginning_video_size_casting

from ... import ROOT_LOGGER_NAME, ROOT_LOGGER_LEVEL
logger = logging.getLogger('{}.{}'.format(ROOT_LOGGER_NAME, __name__))
logger.setLevel(ROOT_LOGGER_LEVEL)


class RGBImageFromFile(BasePreprocessor):
    """
    Provides a transformed RGB image in CHW layout from file
    """
    trial_data = os.path.join(TRIAL_DATA_DIR, 'trial_img.jpg')

    def __init__(self, image_transformer=None, norm=True, *args, **kwargs):
        super(RGBImageFromFile, self).__init__(*args, **kwargs)
        self._transform = image_transformer or (lambda x: x)

        self._shape = self._shape or self.process(self.trial_data).shape
        self._name = self._name or 'default'
        self._norm = norm

    def process(self, data):
        rgb = cv2.cvtColor(cv2.imread(data), cv2.COLOR_BGR2RGB)
        img = self._transform(rgb)

        if self._norm:
            return img.transpose(2, 0, 1) / float(255)
        else:
            return img.transpose(2, 0, 1)

    @property
    def provide_data(self):
        return self._name, self._shape


class RGBImagesFromList(BasePreprocessor):
    """
    Provides a transformed RGB images in CTHW, TCHW layout from list of files
    """
    interpolation_func = {
        'as_is': lambda num_frames_param, num_frames: np.linspace(0, num_frames, num_frames,
                                                                  endpoint=False).astype(int),
        'interpolate': lambda num_frames_param, num_frames: np.linspace(0, num_frames, num_frames_param,
                                                                        endpoint=False).astype(int),
        'loop': loop_video_size_casting,
        'back_and_forth': back_and_fourth_video_size_casting,
        'random_beginning': make_random_beginning_video_size_casting(step=2),
    }

    trial_data = [os.path.join(TRIAL_DATA_DIR, 'trial_img.jpg')]

    def __init__(self, num_frames, mode='interpolate', seq_transformer=None, norm=True, layout='CTHW', *args, **kwargs):
        self._num_frames = num_frames
        self._interpolation = mode
        self._norm = norm
        self._layout = layout

        super(RGBImagesFromList, self).__init__(*args, **kwargs)
        self._transform = seq_transformer or (lambda x: x)

        self._shape = self._shape or RGBImagesFromList.process(self, self.trial_data).shape
        self._name = self._name or 'default'

    def get_image_array(self, data):
        img_arr = []
        for im_file in data:
            img = cv2.cvtColor(cv2.imread(im_file), cv2.COLOR_BGR2RGB)
            img_arr.append(img)

        return img_arr

    def process(self, data):
        inp_img_arr = self.get_image_array(data)

        out_img_arr = self._transform(inp_img_arr)
        time_slice = self.interpolation_func[self._interpolation](self._num_frames, len(out_img_arr))

        img_arr = np.array(out_img_arr)[time_slice, :]

        if self._norm:
            if img_arr.dtype == np.uint8:
                img_arr = img_arr.astype(int)
                img_arr = (img_arr - 128) / float(128)
            else:
                logger.debug('refusing to normalize float array')

        if self._layout == 'CTHW':
            img_arr = img_arr.transpose(3, 0, 1, 2)
        else:
            img_arr = img_arr.transpose(0, 3, 1, 2)

        logger.debug('processed video stats: min={}, max={}'.format(img_arr.min(), img_arr.max()))
        return img_arr

    @property
    def provide_data(self):
        return self._name, self._shape


class RGBImagesFromCallable(RGBImagesFromList):
    def __init__(self, func, num_frames, mode='interpolate', seq_transformer=None, norm=True,
                 layout='CTHW', *args, **kwargs):
        """
        Is buggy when inferring shape
        """
        self._getter = func
        super(RGBImagesFromCallable, self).__init__(num_frames, mode, seq_transformer, norm, layout, *args, **kwargs)

    def get_image_array(self, data):
        return self._getter(data)


class RGBImagesFromData(RGBImagesFromList):
    def __init__(self, format_string,  *args, **kwargs):
        super(RGBImagesFromData, self).__init__(*args, **kwargs)
        self._format_string = format_string

    def get_image_array(self, data):
        arr = np.load(self._format_string.format(data))
        return arr
