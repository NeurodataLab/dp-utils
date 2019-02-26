### Another data processing toolbox

#### Puzzled data iterators
They are composed with index sampler, data processor and iterator whose role to do data processing in parallel. 

```python
import numpy as np
import pandas as pd

from package.data_iterators.iterators.multiprocess_iterator import MultiProcessIterator
from package.data_iterators.balancers.ohc_balancer import OHCBalancer

from package.data_iterators.preprocessors.base_preprocessor import SlowZeroArrayReader
from package.data_iterators.preprocessors.base_preprocessor import IdentityPreprocessor
from package.routines.data_structure_routines import merge_dicts

if __name__ == '__main__':

    labels_data = np.repeat(np.arange(6), axis=0, repeats=1000)
    labels_data = pd.get_dummies(labels_data).values.astype(float)
    balancer = OHCBalancer(data=labels_data, raise_on_end=True)

    data_processor = {
        'data': SlowZeroArrayReader(name='data', shape=(1000, 1000), dtype=float)
    }

    label_processor = {
        'label': IdentityPreprocessor(name='label', shape=(6,))
    }

    iter_train = MultiProcessIterator(
        balancer=balancer, data={'data': labels_data, 'label': labels_data},
        preprocessors=merge_dicts(data_processor, label_processor), use_shared=False,
        batch_size=32, num_processes=6, max_tasks=500, max_results=500
    )

    for num, batch in enumerate(iter_train):
        pass
```

#### Other helper utils
* ffmpeg wrappers
* file utils
* routines, like recursive `mkdir`, multiprocessing `queue` interface built on arrays ... 