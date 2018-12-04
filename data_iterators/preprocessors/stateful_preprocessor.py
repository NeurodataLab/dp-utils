from kungfutils.data_iterators.preprocessors.image_preprocessor import RGBImageFromFile
from .base_preprocessor import BasePreprocessor


class FutureData(object):
    def __init__(self, producer, data_type, queue_index=0):
        """
        :param producer: Parent preprocessor
        :param data_type: 'data' | 'label'
        """
        self._producer = producer
        self._type = data_type
        self._queue_index = queue_index

    def eval(self):
        data_pack = self._producer.process_gather()


class BaseStatefulPreprocessor(BasePreprocessor):
    """
    Base preprocessor for data dependent label preprocessing,
    process data is invoked first, you can save state during data processing, apply this state in process label

    """
    def __init__(self, data_names=None, data_shapes=None, label_names=None, label_shapes=None, *args, **kwargs):
        """
        Ignoring data_shape, label_shape, data_name, label_name
        :param data_names: [list of data names]
        :param data_shapes: [list of data shapes]
        :param label_names: [list of label names]
        :param label_shapes: [list of label shapes]
        """
        super(BaseStatefulPreprocessor, self).__init__()
        self._data_names = data_names
        self._data_shapes = {nm: shape for nm, shape in zip(data_names, data_shapes)}

        self._data_names.extend(label_names)
        self._data_shapes.update({nm: shape for nm, shape in zip(label_names, label_shapes)})

    def provide_data(self, name, *args, **kwargs):
        return name, self._data_shapes[name]

    def process(self, data, *args, **kwargs):
        # name is either named arg or second argument - name = kwargs.get('name', args[0])
        # encode dependencies here
        pass


class DetectionPreprocessor(BaseStatefulPreprocessor):
    layouts_signatures = {'CHW': (2, 0, 1), 'HWC': (0, 1, 2), 'WHC': (0, 2, 1)}

    def __init__(self, image_getter, size_sampler, ratio_sampler, image_aug, norm=True, layout='CHW', *args, **kwargs):
        """
        :param image_getter: image getter, function
        :param size_sampler: function that yields random number in range
        :param image_aug:
        :param norm:
        :param layout:
        :param args:
        :param kwargs:
        """
        super(DetectionPreprocessor, self).__init__(*args, **kwargs)
        self._norm = norm
        self._getter = image_getter
        self._layout = layout
        self._size_sampler = size_sampler

        self._data_proc_meta = {}

    def get_image_array(self, data):
        img = self._getter(data)
        return img

    def process(self, data, *args, **kwargs):
        pass