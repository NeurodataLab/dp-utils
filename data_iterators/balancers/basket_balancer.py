"""
NeurodataLab LLC 21.01.2019
Created by Andrey Belyaev
"""
import numpy as np
import logging

from .base_balancer import BaseBalancer

from ... import ROOT_LOGGER_NAME, ROOT_LOGGER_LEVEL
logger = logging.getLogger('{}.{}'.format(ROOT_LOGGER_NAME, __name__))
logger.setLevel(ROOT_LOGGER_LEVEL)


class BasketBalancer(BaseBalancer):
    """
    Balances samples across baskets
    """
    def __init__(self, data, baskets, raise_on_data_end, raise_on_basket_end, shuffle=True, *args, **kwargs):
        self._baskets = baskets
        self._raise_on_data_end = raise_on_data_end
        self._raise_on_basket_end = raise_on_basket_end
        self._basket_shuffle = shuffle

        self._idxs_per_baskets = {}
        for i, b in enumerate(baskets):
            if b not in self._idxs_per_baskets:
                self._idxs_per_baskets[b] = []
            self._idxs_per_baskets[b].append(i)
        self._idxs_per_baskets = {k: np.asarray(v) for k, v in self._idxs_per_baskets.items()}
        self._baskets_names = self._idxs_per_baskets.keys()

        self._cur_basket_idx = 0
        self._cur_sample_idxs = {i: 0 for i in range(len(self._baskets_names))}
        self._visited = set()

        super(BasketBalancer, self).__init__(data=data, raise_on_end=raise_on_data_end, shuffle=False,
                                             *args, **kwargs)

    def _reset(self, only_cur_basket=False):
        if only_cur_basket:
            if self._basket_shuffle:
                basket_name = self._baskets_names[self._cur_basket_idx]
                basket_idxs = self._idxs_per_baskets[basket_name]
                self._idxs_per_baskets[basket_name] = basket_idxs[np.random.permutation(len(basket_idxs))]
            self._cur_sample_idxs[self._cur_basket_idx] = 0
        else:
            if self._basket_shuffle:
                self._idxs_per_baskets = {k: v[np.random.permutation(len(v))]
                                          for k, v in self._idxs_per_baskets.items()}

            self._cur_basket_idx = 0
            self._cur_sample_idxs = {i: 0 for i in range(len(self._baskets_names))}
            self._visited = set()

    def pre_next(self):
        if self._verbose and len(self._visited) % 100 == 0:
            logger.info("visited set length - {}".format(len(self._visited)))
        if len(self._visited) == self.data_length:
            if self._raise_on_data_end:
                raise StopIteration
            self._reset()

        cur_basket_len = len(self._idxs_per_baskets[self._baskets_names[self._cur_basket_idx]])
        if self._cur_sample_idxs[self._cur_basket_idx] >= cur_basket_len:
            if self._raise_on_basket_end:
                raise StopIteration
            self._reset(only_cur_basket=True)

    @property
    def current_id(self):
        cur_basket_name = self._baskets_names[self._cur_basket_idx]
        cur_sample_idx = self._cur_sample_idxs[self._cur_basket_idx]
        cur_id = self._idxs_per_baskets[cur_basket_name][cur_sample_idx]

        self._cur_sample_idxs[self._cur_basket_idx] += 1
        self._cur_basket_idx = (self._cur_basket_idx + 1) % len(self._baskets_names)

        return cur_id

    def next(self):
        self.pre_next()
        ret_idx = self.current_id
        self.post_next()
        return self._data[ret_idx]
