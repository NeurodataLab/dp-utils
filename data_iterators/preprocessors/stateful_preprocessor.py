class BaseStatefulPreprocessor(object):
    """
    Base preprocessor for data dependent label preprocessing,
    process data is invoked first, you can save state during data processing, apply this state in process label

    """
    def __init__(self, data_name=None, data_shape=None, label_name=None, label_shape=None, *args, **kwargs):
        self._data_name = data_name
        self._data_shape = data_shape

        self._label_name = label_name
        self._label_shape = label_shape

    @property
    def provide_data(self):
        return

    @property
    def provide_label(self):
        return

    def process_data(self, data):
        pass

    def process_label(self, data):
        pass


class DetectionPreprocessor(BaseStatefulPreprocessor):
    layouts_signatures = {'CHW': (2, 0, 1), 'HWC': (0, 1, 2), 'WHC': (0, 2, 1)}

    def __init__(self, image_getter, norm=True, layout='CHW', *args, **kwargs):
        super(DetectionPreprocessor, self).__init__(*args, **kwargs)
        self._norm = norm
        self._getter = image_getter

    @property
    def provide_label(self):
        return self._label_name, self._label_shape

    @property
    def provide_data(self):
        return self._data_name, self._data_shape

    def process_label(self, data):
        pass

    def get_image_array(self, data):
        img = self._getter(data)
        return img

    def process_data(self, data):
        pass
