import numpy as np
import pandas as pd

from kungfutils.data_iterators.iterators.multiprocess_iterator import MultiProcessIterator
from kungfutils.data_iterators.balancers.ohc_balancer import OHCBalancer

from kungfutils.data_iterators.preprocessors.base_preprocessor import ZeroArrayReader
from kungfutils.data_iterators.preprocessors.base_preprocessor import IdentityPreprocessor
from kungfutils.routines.data_structure_routines import merge_dicts

if __name__ == '__main__':

    labels_data = np.repeat(np.arange(6), axis=0, repeats=1000)
    labels_data = pd.get_dummies(labels_data).values
    balancer = OHCBalancer(data=labels_data, raise_on_end=True)

    data_proc = {
        'data': ZeroArrayReader(name='data', shape=(200, 200), dtype=float)
    }

    label_proc = {
        'label': IdentityPreprocessor(name='label', shape=(6,))
    }

    iter_train = MultiProcessIterator(
        balancer=balancer, data={'data': labels_data, 'label': labels_data},
        preprocessors=merge_dicts(data_proc, label_proc),
        batch_size=32,
        num_processes=4,
        max_tasks=500
    )

    while True:
        kek = iter_train.next()
        print(kek)

