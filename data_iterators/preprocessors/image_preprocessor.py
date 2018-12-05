import cv2
import logging

import numpy as np
import os

from .base_preprocessor import BasePreprocessor
from ...data_iterators import TRIAL_DATA_DIR
from ...transformers.resizing import loop_video_size_casting, back_and_fourth_video_size_casting, \
    make_random_beginning_video_size_casting

from ... import ROOT_LOGGER_NAME, ROOT_LOGGER_LEVEL
logger = logging.getLogger('{}.{}'.format(ROOT_LOGGER_NAME, __name__))
logger.setLevel(ROOT_LOGGER_LEVEL)


class RGBImageFromFile(BasePreprocessor):
    """
    Provides a transformed RGB image in CHW layout from file
    """
    trial_data = os.path.join(TRIAL_DATA_DIR, 'trial_img.jpg')
    layouts_signatures = {'CHW': (2, 0, 1), 'HWC': (0, 1, 2), 'WHC': (0, 2, 1)}

    def __init__(self, image_transformer=None, layout='CHW', norm_mean=(0, 0, 0), norm_std=(1, 1, 1), *args, **kwargs):
        super(RGBImageFromFile, self).__init__(*args, **kwargs)
        self._transform = image_transformer or (lambda x: x)

        self._shape = self._shape
        self._name = self._name or 'default'
        self._layout = layout

        self._norm_mean = np.array(norm_mean, dtype=float)
        self._norm_std = np.array(norm_std, dtype=float)

    def process(self, **kwargs):
        processed = {}
        for key, data in kwargs.items():
            rgb = self.get_image_array(data)
            img = self._transform(rgb)

            img = (img - self._norm_mean) / self._norm_std
            img = img.transpose(*self.layouts_signatures[self._layout])
            processed[key] = img
        return processed

    def get_image_array(self, data):
        rgb = cv2.cvtColor(cv2.imread(data), cv2.COLOR_BGR2RGB)
        return rgb


class RGBImageFromCallable(RGBImageFromFile):
    def __init__(self, func, *args, **kwargs):
        """
        Is buggy when inferring shape
        """
        self._getter = func
        super(RGBImageFromCallable, self).__init__(*args, **kwargs)

    def get_image_array(self, data):
        return self._getter(data)


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

    layouts_signatures = {'CTHW': (3, 0, 1, 2), 'TCHW': (0, 3, 1, 2)}

    trial_data = [os.path.join(TRIAL_DATA_DIR, 'trial_img.jpg')]

    def __init__(self, num_frames, mode='interpolate', seq_transformer=None, layout='CTHW',
                 norm_mean=(0, 0, 0), norm_std=(1, 1, 1), *args, **kwargs):
        self._num_frames = num_frames
        self._interpolation = mode
        self._layout = layout

        self._norm_mean = np.array(norm_mean, dtype=float)
        self._norm_std = np.array(norm_std, dtype=float)

        super(RGBImagesFromList, self).__init__(*args, **kwargs)
        self._transform = seq_transformer or (lambda x: x)

        self._shape = self._shape
        self._name = self._name or 'default'

    def get_image_array(self, data):
        img_arr = []
        for im_file in data:
            img = cv2.cvtColor(cv2.imread(im_file), cv2.COLOR_BGR2RGB)
            img_arr.append(img)

        return img_arr

    def process(self, **kwargs):
        processed = {}
        for key, data in kwargs.items():
            inp_img_arr = self.get_image_array(data)

            out_img_arr = self._transform(inp_img_arr)
            time_slice = self.interpolation_func[self._interpolation](self._num_frames, len(out_img_arr))
            img_arr = np.array(out_img_arr)[time_slice, :]

            img_arr = (img_arr - self._norm_mean) / self._norm_std
            img_arr = img_arr.transpose(*self.layouts_signatures[self._layout])

            processed[key] = img_arr
        return processed


class RGBImagesFromCallable(RGBImagesFromList):
    def __init__(self, func, *args, **kwargs):
        self._getter = func
        super(RGBImagesFromCallable, self).__init__(*args, **kwargs)

    def get_image_array(self, data):
        return self._getter(data)


class RGBImagesFromData(RGBImagesFromList):
    def __init__(self, format_string,  *args, **kwargs):
        super(RGBImagesFromData, self).__init__(*args, **kwargs)
        self._format_string = format_string

    def get_image_array(self, data):
        arr = np.load(self._format_string.format(data))
        return arr
