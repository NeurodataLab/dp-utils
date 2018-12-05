from .base_preprocessor import BasePreprocessor
from ...routines.data_structure_routines import merge_dicts


class CompositePreprocessor(BasePreprocessor):
    def __init__(self, data_names, preprocessors, data_shapes, input_names, *args, **kwargs):
        super(CompositePreprocessor, self).__init__(*args, **kwargs)
        self._shapes = data_shapes
        self._names = data_names

        self._input_names = input_names
        self._preprocessors = preprocessors
        self._proc_input_names = {k: processor.provide_input for k, processor in self._preprocessors.items()}

    @property
    def provide_data(self):
        return zip(self._names, self._shapes)

    @property
    def provide_input(self):
        return self._names

    def process(self, **kwargs):
        pass


class DetectionPreprocessor(CompositePreprocessor):
    def __init__(self, preprocessors, *args, **kwargs):
        """
        :param preprocessors: there must be image_getter, label_getter,
                                            image_augmenter, box_crop
        """
        super(DetectionPreprocessor, self).__init__(preprocessors=preprocessors, *args, **kwargs)

    def process(self, **kwargs):
        image_getter_inp = {name: kwargs[name] for name in self._proc_input_names['image_getter']}
        image_getter_out = self._preprocessors['image_getter'].process(**image_getter_inp)

        label_getter_inp = {name: kwargs[name] for name in self._proc_input_names['label_getter']}
        label_getter_out = self._preprocessors['label_getter'].process(**label_getter_inp)

        box_crop_inp_full = merge_dicts(image_getter_out, label_getter_out)
        box_crop_inp = {name: box_crop_inp_full[name] for name in self._proc_input_names['box_crop']}
        box_crop_out = self._preprocessors['box_crop'].process(**box_crop_inp)

        image_aug_inp_full = box_crop_out
        image_aug_inp = {name: image_aug_inp_full[name] for name in self._proc_input_names['image_augmenter']}
        image_aug_out = self._preprocessors['image_augmenter'].process(**image_aug_inp)

        return {k: v for k, v in merge_dicts(image_aug_out, box_crop_out) if k in self._names}