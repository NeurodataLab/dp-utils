import logging
import inspect
import types
from functools import wraps

from .. import ROOT_LOGGER_NAME, ROOT_LOGGER_LEVEL
logger = logging.getLogger('{}.{}'.format(ROOT_LOGGER_NAME, __name__))
logger.setLevel(ROOT_LOGGER_LEVEL)


def autoinit(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if func.__name__ != '__init__':
            return func(self, *args, **kwargs)

        func_spec = inspect.getargspec(func)

        if func_spec.args[0] != 'self':
            raise Exception

        args_len = len(func_spec.args[1:])
        def_len = len(func_spec.defaults)

        for index, name in enumerate(func_spec.args[1:]):
            if name in kwargs:
                val = kwargs[name]
            elif len(args) > index:
                val = args[index]
            elif index >= args_len - def_len:
                val = func_spec.defaults[index - (args_len - def_len)]
            else:
                raise ValueError
            setattr(self, name, val)
        return func(self, *args, **kwargs)
    return wrapper


def fix_documentation(cls):
    for name, func in vars(cls).items():
        if isinstance(func, types.FunctionType) and not func.__doc__:
            logger.info('{} needs documentation'.format(func))
            for parent in cls.__bases__:
                parent_func = getattr(parent, name, None)
                if parent_func and getattr(parent_func, '__doc__', None):
                    func.__doc__ = parent_func.__doc__
                    break
    return cls
