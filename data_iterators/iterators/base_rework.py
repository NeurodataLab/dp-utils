import mxnet as mx
import numpy as np
import logging
from collections import defaultdict

from ...routines.data_structure_routines import merge_dicts
from ... import ROOT_LOGGER_NAME, ROOT_LOGGER_LEVEL
logger = logging.getLogger('{}.{}'.format(ROOT_LOGGER_NAME, __name__))
logger.setLevel(ROOT_LOGGER_LEVEL)


class BaseIterator(object):

    packers = {
        'mxnet': mx.nd.array, 'numpy': np.array, 'list': (lambda x: x)
    }

    def __init__(self, balancer, data, preprocessors, label=None, packers=None,
                 batch_size=32, num_batches=None, return_indices=False):
        """
        :param data: dict {data_name: iterable ...}
        :param label: dict {label_name: iterable ...}
        :param balancer: instance of balancer
        :param preprocessors: dict {(names | name: preprocessor)}
        :param packers: dict {name: 'mxnet' | 'numpy' | 'list'}
        """
        self._return_indices = return_indices

        self._balancer = balancer

        self._label = label or {}
        self._data = data

        self._joint_storage = merge_dicts(self._data, self._label)

        self._preprocessors = preprocessors
        self._one_to_multiple_key_map = {
            k_one: k_mul for k_one in self._joint_keys for k_mul in self._preprocessors.keys() if k_one in k_mul
        }

        self._batch_size = batch_size
        self._num_batches = num_batches
        self._batch_counter = 0

        self._data_keys = list(self._data.keys())
        self._label_keys = list(self._label.keys())
        self._joint_keys = self._data_keys + self._label_keys
        self._packers = packers or {k: 'mxnet' for k in self._joint_keys}
        self._check_packers()

    def __iter__(self):
        return self

    def reset(self):
        self._balancer.reset()

    def add_batch_size(self, provide_product):
        return provide_product[0], (self._batch_size,) + provide_product[1]

    def get_params(self):
        return {
            'batch_size': self.batch_size, 'data': self.provide_data, 'label': self.provide_label,
            'processors': {k: str(v) for k, v in self._preprocessors.items()}
        }

    @property
    def return_indices(self):
        return self._return_indices

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def provide_data(self):
        return [self.add_batch_size(self._preprocessors[key].provide_data(key)) for key in self._data_keys]

    @property
    def provide_label(self):
        return [self.add_batch_size(self._preprocessors[key].provide_data(key)) for key in self._label_keys]

    def _check_packers(self):
        if 'mxnet' in self._packers.values():
            logger.warning('if mxnet packer is set for at least one kind of data, it must be set for every')

    def next(self):
        if self._num_batches is not None and self._num_batches == self._batch_counter:
            raise StopIteration

        data_packs = defaultdict(list)
        indices_to_ret = []
        sample_num = 0
        while sample_num < self._batch_size:
            cur_idx = self._balancer.next()
            instance = None
            try:
                data_instances_to_app = {}
                for key, processor in self._preprocessors.items():
                    key = key if isinstance(key, tuple) else (key,)
                    instance = {k: self._joint_storage[k][cur_idx] for k in key}
                    data_instances_to_app.update(processor.process(**instance))

                sample_num += 1
                indices_to_ret.append(cur_idx)
                for key, data in data_instances_to_app.items():
                    data_packs[key].append(data)
            except (IndexError, IOError) as _:
                logger.info('Probably no data for {}, {}'.format(cur_idx, instance))

        self._batch_counter += 1

        return self._pack_to_backend(data_packs, indices_to_ret)

    __next__ = next

    def _pack_to_backend(self, data_pack, indices_pack):
        data_batched = {key: self.packers[self._packers[key]](data_pack[key]) for key in self._data_keys}
        labels_batched = {key: self.packers[self._packers[key]](data_pack[key]) for key in self._label_keys}

        use_mxnet = 'mxnet' in self._packers.values()

        if use_mxnet:
            logger.info('trying to pack data and labels to mxnet batch')
            if not self._return_indices:
                return mx.io.DataBatch(data=data_batched, label=labels_batched, pad=0)
            else:
                return mx.io.DataBatch(data=data_batched, label=labels_batched, pad=0), indices_pack
        else:
            if not self._return_indices:
                return data_batched, labels_batched
            else:
                return (data_batched, labels_batched), indices_pack


