import numpy as np
import logging

from base_balancer import BaseBalancer


class OHCBalancer(BaseBalancer):
    """
    Balances equalizing classes by OH data
    """
    def __init__(self, data, raise_on_end, shuffle=True, *args, **kwargs):
        """
        :param data: numpy array with (num_samples, num_classes) shape
        :param raise_on_end: either raise when everything is visited
        """
        super(OHCBalancer, self).__init__(data=data, raise_on_end=raise_on_end, shuffle=shuffle, *args, **kwargs)

    def _reset(self):
        self._current_class = 0
        self._num_classes = self._data.shape[1]

        if self._shuffle:
            self._perm = np.random.permutation(self.data_length)
        else:
            self._perm = np.arange(self.data_length, dtype=int)

        self._visited = set()

        self._split_by_classes()

    def _split_by_classes(self):
        self._per_class_index = []

        row_ids, col_ids = np.where(self._data[self._perm] == 1)
        for class_id in range(self._num_classes):
            self._per_class_index.append(row_ids[col_ids == class_id])
        self._per_class_current_pointers = [0 for _ in self._per_class_index]

    def _increment_current_class_pointer(self):
        cur_val = self.current_in_class_pointer + 1
        if cur_val == len(self._per_class_index[self.current_class]):
            self.current_in_class_pointer = 0
        else:
            self.current_in_class_pointer += 1

    def _increment_current_class(self):
        cur_val = self.current_class + 1
        if cur_val == self._num_classes:
            self.current_class = 0
        else:
            self.current_class += 1

    def pre_next(self):
        if len(self._visited) % 100 == 0:
            logging.getLogger().info("visited set length - {}".format(len(self._visited)))
        if len(self._visited) == self.data_length:
            self.reset()
            if self._raise_on_end:
                raise StopIteration

    def post_next(self):
        self._increment_current_class_pointer()
        self._increment_current_class()

    @property
    def current_class(self):
        return self._current_class

    @current_class.setter
    def current_class(self, value):
        self._current_class = value

    @property
    def current_in_class_pointer(self):
        return self._per_class_current_pointers[self.current_class]

    @current_in_class_pointer.setter
    def current_in_class_pointer(self, value):
        self._per_class_current_pointers[self.current_class] = value

    @property
    def data_length(self):
        return self._data.shape[0]

    @property
    def current_id(self):
        ret_val = self._per_class_index[self.current_class][self.current_in_class_pointer]
        self._visited.add(ret_val)
        return ret_val

    def next(self):
        self.pre_next()
        ret_idx = self.current_id
        self.post_next()
        return self._perm[ret_idx]


