import mxnet as mx
import logging

from ... import ROOT_LOGGER_NAME, ROOT_LOGGER_LEVEL
logger = logging.getLogger('{}.{}'.format(ROOT_LOGGER_NAME, __name__))
logger.setLevel(ROOT_LOGGER_LEVEL)


class BaseIterator(object):

    def __init__(self, balancer, data, data_preprocessors, label=None, return_indices=False,
                 label_preprocessors=None, batch_size=32, num_batches=None):
        """
        :param data: dict {data_name: iterable ...}
        :param label: dict {label_name: iterable ...}
        :param balancer: instance of balancer
        :param data_preprocessors: dict {data_name: preprocessor}
        :param label_preprocessors: dict {label_name: preprocessor}
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

        self._init_batch_storage()

    def __iter__(self):
        return self

    def _init_batch_storage(self):
        self._batch_storage_data = []
        self._batch_storage_label = []

    def reset(self):
        self._balancer.reset()
        self._init_batch_storage()

    def add_batch_size(self, provide_product):
        return provide_product[0], (self._batch_size,) + provide_product[1]

    @property
    def provide_data(self):
        return [self.add_batch_size(self._data_preprocessors[key].provide_data) for key in self._data_keys]

    @property
    def provide_label(self):
        return [self.add_batch_size(self._label_preprocessors[key].provide_data) for key in self._label_keys]

    @staticmethod
    def process_sample(processors, data, storage):
        """
        This function is applied per batch
        :param processors: list
        :param data: list
        :param storage: some storage, here list
        """
        storage.append([proc.process(d) for d, proc in zip(data, processors)])

    @staticmethod
    def collect_samples(storage):
        """
        :param storage: some storage, here list
        :return: data list (batch size, number of data), label_list (batch size, number of labels)
        """
        return storage

    def next(self):
        if self._num_batches is not None and self._num_batches == self._batch_counter:
            raise StopIteration

        data_processors = [self._data_preprocessors[key] for key in self._data_keys]
        label_processors = [self._label_preprocessors[key] for key in self._label_keys]

        indices_to_ret = []

        sample_num = 0
        while sample_num < self._batch_size:
            cur_idx = self._balancer.next()
            data = [self._data[key][cur_idx] for key in self._data_keys]
            labels = [self._label[key][cur_idx] for key in self._label_keys]

            try:
                self.process_sample(
                    processors=data_processors, data=data, storage=self._batch_storage_data
                )
                self.process_sample(
                    processors=label_processors, data=labels, storage=self._batch_storage_label
                )
                sample_num += 1
                indices_to_ret.append(cur_idx)
            except (IndexError, IOError) as _:
                logger.info('probably no data for {}, data: {}, label: {}'.format(cur_idx, data, labels))

        data_packs = self.collect_samples(self._batch_storage_data)
        label_packs = self.collect_samples(self._batch_storage_data)

        data_batched = [mx.nd.array(data_pack) for data_pack in data_packs]
        labels_batched = [mx.nd.array(label_pack) for label_pack in label_packs]

        self._batch_counter += 1
        if not self._return_indices:
            return mx.io.DataBatch(data=data_batched, label=labels_batched, pad=0)
        else:
            return mx.io.DataBatch(data=data_batched, label=labels_batched, pad=0), indices_to_ret
