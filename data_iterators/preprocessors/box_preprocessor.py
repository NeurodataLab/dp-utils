from .base_preprocessor import MIMOProcessor


class BoxLabelGetter(MIMOProcessor):
    def __init__(self, func, data_names=('boxes', 'labels'), input_names=('label',), *args, **kwargs):
        super(BoxLabelGetter, self).__init__(data_names=data_names, input_names=input_names, *args, **kwargs)
        self._getter = func

    def process(self, **kwargs):
        boxes, labels = self._getter(kwargs[self.provide_input[0]])
        return {self.provide_output[0]: boxes, self.provide_output[1]: labels}


class BoxLabelProcessor(MIMOProcessor):
    def __init__(self, func, input_names=('boxes', 'labels'), *args, **kwargs):
        super(BoxLabelProcessor, self).__init__(input_names=input_names, *args, **kwargs)
        self._transformer = func

    def process(self, **kwargs):
        bundle = self._transformer(**kwargs)
        return {name: data for name, data in zip(self.provide_output, bundle)}


class CropRGBImage(MIMOProcessor):
    def __init__(self, data_names=('image',), input_names=('image', 'crop'), *args, **kwargs):
        super(CropRGBImage, self).__init__(data_names=data_names, input_names=input_names, *args, **kwargs)

    def process(self, **kwargs):
        # crop is x, y, x, y, image is hwc
        image, crop = kwargs[self.provide_input[0]], kwargs[self.provide_input[1]]
        cropped = image[slice(*crop[1::2]), slice(*crop[0::2])]
        return {self.provide_output[0]: cropped}