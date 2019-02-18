import mxnet as mx
import numpy as np
import logging
from collections import defaultdict

from ... import ROOT_LOGGER_NAME, ROOT_LOGGER_LEVEL
logger = logging.getLogger('{}.{}'.format(ROOT_LOGGER_NAME, __name__))
logger.setLevel(ROOT_LOGGER_LEVEL)


class BaseIterator(object):

    packers = {
        'mxnet': mx.nd.array, 'numpy': np.array, 'list': (lambda x: x)
    }

    def __init__(self, balancer, data, preprocessors, packers=None,
                 batch_size=32, num_batches=None, return_indices=False):
        """
        :param data: dict {data_name: iterable ...}
        :param balancer: instance of balancer
        :param preprocessors: dict {(names | name: preprocessor)}
        :param packers: dict {name: 'mxnet' | 'numpy' | 'list'}
        """
        self._return_indices = return_indices
        self._balancer = balancer

        self._data = data
        self._preprocessors = preprocessors

        self._batch_size = batch_size
        self._num_batches = num_batches
        self._batch_counter = 0

        self._packers = packers or {name: 'mxnet' for name, shape in self.provide_data}
        self._check_packers()

    def __iter__(self):
        return self

    def reset(self):
        self._balancer.reset()

    def add_batch_size(self, provide_product):
        return provide_product[0], (self._batch_size,) + provide_product[1]

    def get_params(self):
        return {
            'batch_size': self.batch_size, 'data': self.provide_data,
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
        return [
            self.add_batch_size(provided) for processor in self._preprocessors.values()
            for provided in processor.provide_data
        ]

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
                    input_keys = processor.provide_input
                    instance = {k: self._data[k][cur_idx] for k in input_keys}
                    data_instances_to_app.update(processor.process(**instance))

                sample_num += 1
                indices_to_ret.append(cur_idx)
                for key, data in data_instances_to_app.items():
                    data_packs[key].append(data)
            except (IndexError, IOError, ValueError) as _:
                logger.info('Probably no data for {}, {}'.format(cur_idx, instance))

        self._batch_counter += 1

        return self._pack_to_backend(data_packs, indices_to_ret)

    __next__ = next

    def _pack_to_backend(self, data_pack, indices_pack):
        data_batched = [self.packers[self._packers[key]](data_pack[key]) for key, _ in self.provide_data]
        if not self._return_indices:
            return data_batched
        else:
            return data_batched, indices_pack
