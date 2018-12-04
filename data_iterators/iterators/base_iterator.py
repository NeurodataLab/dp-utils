import mxnet as mx
import numpy as np
import logging

from ... import ROOT_LOGGER_NAME, ROOT_LOGGER_LEVEL
logger = logging.getLogger('{}.{}'.format(ROOT_LOGGER_NAME, __name__))
logger.setLevel(ROOT_LOGGER_LEVEL)


class BaseIterator(object):

    packers = {
        'mxnet': mx.nd.array, 'numpy': np.array, 'list': (lambda x: x)
    }

    def __init__(self, balancer, data, data_preprocessors, data_packers=None,
                 label=None, label_preprocessors=None, label_packers=None,
                 batch_size=32, num_batches=None, return_indices=False):
        """
        :param data: dict {data_name: iterable ...}
        :param label: dict {label_name: iterable ...}
        :param balancer: instance of balancer
        :param data_preprocessors: dict {data_name: preprocessor}
        :param label_preprocessors: dict {label_name: preprocessor}
        :param data_packers: dict {data_name: 'mxnet' | 'numpy' | 'list'}
        :param label_packers: dict {label_name: 'mxnet' | 'numpy' | 'list'}
        """
        self._return_indices = return_indices

        self._balancer = balancer

        self._data = data
        self._data_preprocessors = data_preprocessors

        self._label = label or {}
        self._label_preprocessors = label_preprocessors or {}

        self._batch_size = batch_size
        self._num_batches = num_batches
        self._batch_counter = 0

        self._data_keys = list(data.keys())
        self._label_keys = list(label.keys())

        self._data_packers = data_packers or {k: 'mxnet' for k in self._data_keys}
        self._label_packers = label_packers or {k: 'mxnet' for k in self._label_keys}
        self._check_packers()

    def __iter__(self):
        return self

    def reset(self):
        self._balancer.reset()

    def add_batch_size(self, provide_product):
        return provide_product[0], (self._batch_size,) + provide_product[1]

    def get_params(self):
        return {'batch_size': self.batch_size, 'data': self.provide_data, 'label': self.provide_label,
                'data_processors': {k: str(v) for k, v in self._data_preprocessors.items()},
                'label_processors': {k: str(v) for k, v in self._label_preprocessors.items()}
                }

    @property
    def return_indices(self):
        return self._return_indices

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def provide_data(self):
        return [self.add_batch_size(self._data_preprocessors[key].provide_data(key)) for key in self._data_keys]

    @property
    def provide_label(self):
        return [self.add_batch_size(self._label_preprocessors[key].provide_data(key)) for key in self._label_keys]

    def _check_packers(self):
        for packer_pack in [self._data_packers, self._label_packers]:
            if 'mxnet' in packer_pack.values():
                logger.warning('if mxnet packer is set for at least one kind of data, it must be set for every')

    def _pack_to_backend(self, data_packs, label_packs, indices_pack):
        data_batched = [self.packers[self._data_packers[data_key]](data_packs[num])
                        for num, data_key in enumerate(self._data_keys)]
        labels_batched = [self.packers[self._label_packers[label_key]](label_packs[num])
                          for num, label_key in enumerate(self._label_keys)]

        use_mxnet = 'mxnet' in self._data_packers.values()

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

    def next(self):
        if self._num_batches is not None and self._num_batches == self._batch_counter:
            raise StopIteration

        data_packs = [[] for _ in self._data_keys]
        label_packs = [[] for _ in self._label_keys]

        indices_to_ret = []
        sample_num = 0
        while sample_num < self._batch_size:
            cur_idx = self._balancer.next()
            instance = None
            try:
                data_instances_to_app = []
                for num, key in enumerate(self._data_keys):
                    process_func = self._data_preprocessors[key].process

                    instance = self._data[key][cur_idx]
                    to_app = process_func(instance, name=key)
                    data_instances_to_app.append(to_app)

                label_instances_to_app = []
                for num, key in enumerate(self._label_keys):
                    process_func = self._label_preprocessors[key].process

                    instance = self._label[key][cur_idx]
                    to_app = process_func(instance, name=key)
                    label_instances_to_app.append(to_app)
                # no append to batch before possible exception

                for num, to_app in enumerate(data_instances_to_app):
                    data_packs[num].append(to_app)

                for num, to_app in enumerate(label_instances_to_app):
                    label_packs[num].append(to_app)

                sample_num += 1
                indices_to_ret.append(cur_idx)
            except (IndexError, IOError) as _:
                logger.info('probably no data for {}, {}'.format(cur_idx, instance))

        self._batch_counter += 1

        return self._pack_to_backend(data_packs, label_packs, indices_to_ret)

    __next__ = next
