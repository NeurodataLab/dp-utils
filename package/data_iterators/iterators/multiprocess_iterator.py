import multiprocessing as mp
from collections import defaultdict
from queue import Full, Empty

import logging
import numpy as np
from multiprocessing_logging import install_mp_handler

from .base_iterator import BaseIterator
from ...routines.mp_routines import ArrayDictQueue
from ... import ROOT_LOGGER_NAME, ROOT_LOGGER_LEVEL
logger = logging.getLogger('{}.{}'.format(ROOT_LOGGER_NAME, __name__))
logger.setLevel(ROOT_LOGGER_LEVEL)
install_mp_handler(logger=logger)


class MultiProcessIterator(BaseIterator):
    """Iterates through data with base iterator interface, implementing in-batch parallelism"""
    def __init__(self, num_processes, max_tasks=100, max_results=100, use_shared=False, *args, **kwargs):
        """
        :param num_processes: number of processes two be used by iterator
        :param max_tasks: max number of tasks to be put in tasks queue
        :param max_results: max volume of output queue
        :param use_shared: whether to use array queue for passing big arrays without pickling them
        """

        super(MultiProcessIterator, self).__init__(*args, **kwargs)
        self._num_processes = num_processes
        self._max_tasks = max(max_tasks, self._batch_size)
        self._max_results = max(max_results, self._batch_size)

        self._input_storage = mp.Queue(maxsize=self._max_tasks)
        if use_shared:
            check_packers_sh_mem = reduce(
                lambda x, y: x and y, [i in ['numpy', 'mxnet'] for i in self._packers.values()]
            )
            assert check_packers_sh_mem, 'array packers needed for shared memory iterator'
            logger.warning("Only floats are supported for array queue")
            self._output_storage = ArrayDictQueue(
                templates={name: np.zeros(shape[1:], dtype=float) for name, shape in self.provide_data},
                maxsize=self._max_results)
        else:
            self._output_storage = mp.Queue(maxsize=self._max_results)

        worker_func = self._make_worker_func()
        self._workers = []
        for i in range(num_processes):
            proc = mp.Process(target=worker_func,  args=(self._input_storage, self._output_storage))
            proc.daemon = True
            proc.start()
            self._workers.append(proc)
        self._continue = self.next_tasks()

    def _make_worker_func(self):
        def task_func(task_queue, result_queue):
            while True:
                result = {}
                bundle = task_queue.get()  # waiting for available task
                idx, data_pack = bundle['index'], {k: v for k, v in bundle.items() if k != 'index'}
                try:
                    for key, processor in self._preprocessors.items():
                        input_keys = processor.provide_input
                        instance = {k: data_pack[k] for k in input_keys}

                        result.update(processor.process(**instance))
                    result.update({'index': idx})
                    result_queue.put(result)

                except (IndexError, IOError) as _:
                    logger.info('Probably no data for {}'.format(idx))

        return task_func

    def reset(self):
        super(MultiProcessIterator, self).reset()
        self._continue = self.next_tasks()

    def next_tasks(self, num_tasks=None):
        task_added = 0
        num_tasks = num_tasks or self._max_tasks
        while task_added < num_tasks:  # iterate until full or stop
            try:
                idx = self._balancer.next()
                data_pack = {key: data[idx] for key, data in self._data.items()}
                data_pack.update({'index': idx})

                self._input_storage.put_nowait(data_pack)
                task_added += 1
            except Full:
                return True
            except StopIteration:
                return False
        return True

    def next(self):
        if self._num_batches is not None and self._num_batches == self._batch_counter:
            raise StopIteration

        sample_num = 0
        data_packs = defaultdict(list)
        indices_to_ret = []
        while sample_num < self._batch_size:
            try:
                bundle = self._output_storage.get(True, 10)
                idx, data_dict = bundle['index'], {k: v for k, v in bundle.items() if k != 'index'}
                # idx - sample index, data_dict - {'name': data, ...}
                # if no exception here
                sample_num += 1
                indices_to_ret.append(idx)
                for key, data in data_dict.items():
                    data_packs[key].append(data)
            except Empty:
                break

        self._continue = self.next_tasks() if self._continue else self._continue
        if not self._continue and len(indices_to_ret) == 0:
            raise StopIteration

        return self._pack_to_backend(data_packs, indices_to_ret)

