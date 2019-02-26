from multiprocessing import Queue, Array
import numpy as np


class ArrayDictQueue(object):
    def __init__(self, templates, maxsize=0):
        if maxsize == 0:
            # this queue cannot be infinite, because it will be backed by real objects
            raise ValueError('ArrayQueue(template, maxsize) must use a finite value for maxsize.')

        # find the size and data type for the arrays
        # note: every ndarray put on the queue must be this size
        self.dtypes = {name: template.dtype for name, template in templates.items()}
        self.shapes = {name: template.shape for name, template in templates.items()}
        self.byte_counts = {name: template.size * template.itemsize for name, template in templates.items()}

        self.template_names = templates.keys()
        # make a pool of numpy arrays, each backed by shared memory,
        # and create a queue to keep track of which ones are free
        self.array_pool = {name: [None] * maxsize for name in self.template_names}
        self.free_arrays = Queue(maxsize)
        for array_id in range(maxsize):
            for name in self.template_names:
                dtype, shape, byte_count = self.dtypes[name], self.shapes[name], self.byte_counts[name]

                buf = Array('c', byte_count, lock=False)
                self.array_pool[name][array_id] = np.frombuffer(buf, dtype=dtype).reshape(shape)
            self.free_arrays.put(array_id)

        self.q = Queue(maxsize)

    def put(self, items, *args, **kwargs):
        dict_to_put = {}
        # get the ID of an available shared-memory array
        free_id = self.free_arrays.get()

        for name, item in items.items():
            if type(item) is np.ndarray and name in self.template_names:
                dtype, shape, byte_count = self.dtypes[name], self.shapes[name], self.byte_counts[name]

                item_size = item.size * item.itemsize
                if item.dtype == dtype and item.shape == shape and item_size == byte_count:
                    # copy item to the shared-memory array
                    self.array_pool[name][free_id][:] = item
                    # put the array's free_id (not the whole array) onto the queue
                    # dict_to_put[name] = free_id
                else:
                    raise ValueError(
                        'ndarray does not match type or shape of template used to initialize ArrayQueue'
                    )
            else:
                dict_to_put[name] = item
        dict_to_put['__array_id__'] = free_id
        self.q.put(dict_to_put, *args, **kwargs)

    def get(self, *args, **kwargs):
        items = self.q.get(*args, **kwargs)  # *args, **kwargs are important
        ret_items = {}
        for q_name, item in items.items():
            if q_name == '__array_id__':
                for name in self.template_names:
                    # item is the id of a shared-memory array
                    # copy the array
                    arr = self.array_pool[name][item].copy()
                    ret_items[name] = arr
            else:
                ret_items[q_name] = item
        # put the shared-memory array back into the pool
        self.free_arrays.put(items['__array_id__'])

        return ret_items

    def get_nowait(self):
        raise NotImplementedError

    def put_nowait(self):
        raise NotImplementedError
