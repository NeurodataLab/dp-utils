import mxnet as mx


class MXNetBatchWrapper(object):
    def __init__(self, data_names, label_names, iterator, pad=False):
        self._iterator = iterator

        self._data_names = data_names
        self._label_names = label_names

        self._pad_delta = 0
        self._pad = pad

    def _pad_batch(self, data):
        self._pad_delta = self._iterator.batch_size - data.shape[0]

        if self._pad_delta == 0:
            return data
        else:
            indices = mx.nd.arange(0, self._iterator.batch_size, dtype=int)
            indices_zeros = mx.nd.zeros_like(indices)
            indices = mx.nd.where(indices >= data.shape[0], indices_zeros, indices)
            return mx.nd.take(data, indices=indices, axis=0)

    def next(self):
        if self._iterator.return_indices:
            ret_arrays, _ = self._iterator.next()
        else:
            ret_arrays = self._iterator.next()

        whole = self._iterator.provide_data

        if self._pad:
            data = [self._pad_batch(ret_arrays[num]) for num, desc in enumerate(whole) if desc in self.provide_data]
            labels = [self._pad_batch(ret_arrays[num]) for num, desc in enumerate(whole) if desc in self.provide_label]
        else:
            data = [ret_arrays[num] for num, desc in enumerate(whole) if desc in self.provide_data]
            labels = [ret_arrays[num] for num, desc in enumerate(whole) if desc in self.provide_label]

        return mx.io.DataBatch(data=data, label=labels, pad=self._pad_delta)

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

