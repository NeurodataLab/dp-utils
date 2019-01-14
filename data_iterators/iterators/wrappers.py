import mxnet as mx


class MXNetBatchWrapper(object):
    def __init__(self, data_names, label_names, iterator):
        self._iterator = iterator

        self._data_names = data_names
        self._label_names = label_names

    def next(self):
        if self._iterator.return_indices:
            ret_arrays, _ = self._iterator.next()
        else:
            ret_arrays = self._iterator.next()

        whole = self._iterator.provide_data

        data = [ret_arrays[num] for num, desc in enumerate(whole) if desc in self.provide_data]
        labels = [ret_arrays[num] for num, desc in enumerate(whole) if desc in self.provide_label]

        if self._iterator.batch_size - len(data[0]) != 0:
            raise StopIteration
        return mx.io.DataBatch(data=data, label=labels, pad=0)

    def iter(self):
        return self

    def reset(self):
        self._iterator.reset()

    __iter__ = iter

    @property
    def provide_data(self):
        whole = self._iterator.provide_data
        return [(name, shape) for name, shape in whole if name in self._data_names]

    @property
    def provide_label(self):
        whole = self._iterator.provide_data
        return [(name, shape) for name, shape in whole if name in self._label_names]

