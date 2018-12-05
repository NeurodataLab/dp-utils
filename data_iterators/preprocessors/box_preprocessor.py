from .base_preprocessor import BasePreprocessor


class MIMOProcessor(BasePreprocessor):
    def __init__(self, data_names, data_shapes, input_names, *args, **kwargs):
        super(MIMOProcessor, self).__init__(*args, **kwargs)
        self._data_shapes = data_shapes
        self._data_names = data_names

        self._input_names = input_names

    @property
    def provide_input(self):
        return self._input_names

    @property
    def provide_data(self):
        return list(zip(self._data_names, self._data_shapes))


class BoxLabelGetter(MIMOProcessor):
    def __init__(self, func, data_shapes, data_names=('boxes', 'labels'), input_names=('label',), *args, **kwargs):
        super(BoxLabelGetter, self).__init__(*args, **kwargs)
        self._data_shapes = data_shapes
        self._data_names = data_names

        self._input_names = input_names
        self._getter = func

    def process(self, **kwargs):
        boxes, labels = self._getter(kwargs[self.provide_input[0]])
        return {self._data_names[0]: boxes, self._data_names[1]: labels}


class BoxLabelProcessor(MIMOProcessor):
    def __init__(self, func, data_names, data_shapes, input_names=('boxes', 'labels'), *args, **kwargs):
        super(BoxLabelProcessor, self).__init__(*args, **kwargs)
        self._data_shapes = data_shapes
        self._data_names = data_names

        self._input_names = input_names
        self._transformer = func

    def process(self, **kwargs):
        bundle = self._transformer(boxes=kwargs[self.provide_input[0]], labels=kwargs[self.provide_input[1]])
        return {name: data for (name, _), data in zip(self.provide_data, bundle)}


class CropRGBImage(MIMOProcessor):
    def __init__(self, data_shapes, data_names=('image',), input_names=('image', 'crop'), *args, **kwargs):
        super(CropRGBImage, self).__init__(*args, **kwargs)
        self._data_shapes = data_shapes
        self._data_names = data_names

        self._input_names = input_names

    def process(self, **kwargs):
        # crop is x, y, x, y, image is hwc
        image, crop = kwargs[self.provide_input[0]], kwargs[self.provide_input[1]]
        cropped = image[slice(*crop[1::2]), slice(*crop[0::2])]
        return {self._data_names[0]: cropped}