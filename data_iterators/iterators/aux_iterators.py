import numpy as np
import logging

from .base_iterator import BaseIterator

from ... import ROOT_LOGGER_NAME, ROOT_LOGGER_LEVEL
logger = logging.getLogger('{}.{}'.format(ROOT_LOGGER_NAME, __name__))
logger.setLevel(ROOT_LOGGER_LEVEL)


class NumpyBaseIterator(BaseIterator):

    def _pack_to_backend(self, data_packs, label_packs, indices_pack):
        data_batched = [np.array(data_pack) for data_pack in data_packs]
        labels_batched = [np.array(label_pack) for label_pack in label_packs]

        if not self._return_indices:
            return data_batched, labels_batched
        else:
            return (data_batched, labels_batched), indices_pack

# TODO: implement mp numpy version, mp version is not well tested yet
