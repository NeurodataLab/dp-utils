import numpy as np
import logging

from ... import ROOT_LOGGER_NAME, ROOT_LOGGER_LEVEL

logger = logging.getLogger('{}.{}'.format(ROOT_LOGGER_NAME, __name__))
logger.setLevel(ROOT_LOGGER_LEVEL)


class BasePreprocessor(object):

    def __init__(self, name=None, shape=None, *args, **kwargs):
        self._shape = shape
        self._name = name

    def process(self, data):
        pass

    @property
    def provide_data(self):
        return self._name, self._shape


class IdentityPreprocessor(BasePreprocessor):
    def __init__(self, *args, **kwargs):
        super(IdentityPreprocessor, self).__init__(*args, **kwargs)

    def process(self, data):
        return np.atleast_1d(data)


class ArrayReader(BasePreprocessor):
    def __init__(self, name, shape, format_string, *args, **kwargs):
        super(ArrayReader, self).__init__(name, shape, *args, **kwargs)
        self._format_string = format_string

    def process(self, data):
        return np.load(self._format_string.format(data))


class ZeroArrayReader(BasePreprocessor):
    def __init__(self, name=None, shape=None, dtype=float, *args, **kwargs):
        super(ZeroArrayReader, self).__init__(name, shape, *args, **kwargs)
        self._dt = dtype

    def process(self, data):
        return np.zeros(self._shape, dtype=self._dt)


class ArrayGetter(BasePreprocessor):
    def __init__(self, func=None, *args, **kwargs):
        self._getter_func = func
        super(ArrayGetter, self).__init__(*args, **kwargs)

    def process(self, data):
        return self._getter_func(data)


class SlowZeroArrayReader(ZeroArrayReader):
    def process(self, data):
        count_to = 0
        for i in range(10000):
            count_to += 1
        return super(SlowZeroArrayReader, self).process(data)
