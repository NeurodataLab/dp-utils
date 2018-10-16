import numpy as np


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
