import mxnet as mx
import logging

from kungfutils.data_iterators.iterators.base_iterator import BaseIterator
from ... import ROOT_LOGGER_NAME, ROOT_LOGGER_LEVEL
logger = logging.getLogger('{}.{}'.format(ROOT_LOGGER_NAME, __name__))
logger.setLevel(ROOT_LOGGER_LEVEL)


class MultiProcessIterator(BaseIterator):
    def __init__(self, num_processes, *args, **kwargs):
        super(MultiProcessIterator, self).__init__(*args, **kwargs)
        self._num_processes = num_processes


    @staticmethod
    def process_instance(processorinstance):
        pass

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
                for num, key in enumerate(self._data_keys):
                    instance = self._data[key][cur_idx]
                    to_app = self._data_preprocessors[key].process(self._data[key][cur_idx])
                    data_packs[num].append(to_app)

                for num, key in enumerate(self._label_keys):
                    instance = self._label[key][cur_idx]
                    to_app = self._label_preprocessors[key].process(self._label[key][cur_idx])
                    label_packs[num].append(to_app)

                sample_num += 1
                indices_to_ret.append(cur_idx)
            except (IndexError, IOError) as _:
                logger.info('probably no data for {}, {}'.format(cur_idx, instance))

        data_batched = [mx.nd.array(data_pack) for data_pack in data_packs]
        labels_batched = [mx.nd.array(label_pack) for label_pack in label_packs]

        self._batch_counter += 1
        if not self._return_indices:
            return mx.io.DataBatch(data=data_batched, label=labels_batched, pad=0)
        else:
            return mx.io.DataBatch(data=data_batched, label=labels_batched, pad=0), indices_to_ret