import numpy as np
import pandas as pd

from package.data_iterators.iterators.multiprocess_iterator import MultiProcessIterator
from package.data_iterators.samplers.ohc_balancer import OHCBalancer

from package.data_iterators.preprocessors.base_preprocessor import SlowZeroArrayReader
from package.data_iterators.preprocessors.base_preprocessor import IdentityPreprocessor
from package.routines.data_structure_routines import merge_dicts

if __name__ == '__main__':

    labels_data = np.repeat(np.arange(6), axis=0, repeats=10000)
    labels_data = pd.get_dummies(labels_data).values
    balancer = OHCBalancer(data=labels_data, raise_on_end=True)

    data_proc = {
        'data': SlowZeroArrayReader(name='data', shape=(200, 200), dtype=float)
    }

    label_proc = {
        'label': IdentityPreprocessor(name='label', shape=(6,))
    }

    iter_train = MultiProcessIterator(
        balancer=balancer, data={'data': labels_data, 'label': labels_data},
        preprocessors=merge_dicts(data_proc, label_proc),
        batch_size=32,
        num_processes=8,
        max_tasks=5000,
        max_results=5000,
    )

    while True:
        kek = iter_train.next()
        print(kek)

