import numpy as np
import logging

from .ohc_balancer import OHCBalancer

from ... import ROOT_LOGGER_NAME, ROOT_LOGGER_LEVEL
logger = logging.getLogger('{}.{}'.format(ROOT_LOGGER_NAME, __name__))
logger.setLevel(ROOT_LOGGER_LEVEL)


class SoftmaxBalancer(OHCBalancer):
    """
    Balances on argmax instead of one in one hot encoding
    """
    def _split_by_classes(self):
        self._per_class_index = []

        row_ids, col_ids = np.arange(self.data_length, dtype=int), np.argmax(self._data[self._perm], axis=1)
        for class_id in range(self._num_classes):
            self._per_class_index.append(row_ids[col_ids == class_id])
        self._per_class_current_pointers = [0 for _ in self._per_class_index]
