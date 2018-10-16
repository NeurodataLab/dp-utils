import mxnet as mx
import logging


class BaseIterator(object):

    def __init__(self, balancer, data, data_preprocessors, label=None,
                 label_preprocessors=None, batch_size=32, num_batches=None):
        """
        :param data: dict {data_name: iterable ...}
        :param label: dict {label_name: iterable ...}
        :param balancer: instance of balancer
        :param data_preprocessors: dict {data_name: preprocessor}
        :param label_preprocessors: dict {label_name: preprocessor}
        """
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

    def add_batch_size(self, provide_product):
        return provide_product[0], (self._batch_size,) + provide_product[1]

    @property
    def provide_data(self):
        return [self.add_batch_size(self._data_preprocessors[key].provide_data) for key in self._data_keys]

    @property
    def provide_label(self):
        return [self.add_batch_size(self._label_preprocessors[key].provide_data) for key in self._label_keys]

    def __iter__(self):
        return self

    def reset(self):
        self._balancer.reset()

    def next(self):
        if self._num_batches is not None and self._num_batches == self._batch_counter:
            raise StopIteration

        data_packs = [[] for _ in self._data_keys]
        label_packs = [[] for _ in self._label_keys]

        sample_num = 0
        while sample_num < self._batch_size:
            cur_idx = self._balancer.next()
            try:
                for num, key in enumerate(self._data_keys):
                    to_app = self._data_preprocessors[key].process(self._data[key][cur_idx])
                    data_packs[num].append(to_app)

                for num, key in enumerate(self._label_keys):
                    to_app = self._label_preprocessors[key].process(self._label[key][cur_idx])
                    label_packs[num].append(to_app)
                sample_num += 1
            except IndexError:
                logging.getLogger().info('probably no bbox for {}'.format(cur_idx))

        data_batched = [mx.nd.array(data_pack) for data_pack in data_packs]
        labels_batched = [mx.nd.array(label_pack) for label_pack in label_packs]

        self._batch_counter += 1
        return mx.io.DataBatch(data=data_batched, label=labels_batched, pad=0)