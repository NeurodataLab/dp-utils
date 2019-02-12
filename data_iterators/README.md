Iteration process consists of three stages:
* Retrieving index of data instance from balancer (one of the classes from `balancers` package)
* Calling preprocessors for each kind of data: several choices here are possible: 
    1) either you have complex interdependent 
processing, in which case you have to use graph based preprocessor from `preprocessors/composite_preprocessors`, it has 
a Keras-like interface, where you are able to stack additional processors over available graph heads.
    2) or everything is simple as label encoding or image augmentation
* Combining all these things into iterator: `iterators/base_iterator` or `iterators/multiprocess_iterator`

Several helper processors are available in: `preprocessors/image_preprocessor` and `preprocessors/box_preprocessors`