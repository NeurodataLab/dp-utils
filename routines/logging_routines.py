from functools import wraps


def log_args_into(logger):

    def logger_decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger.debug('Passed args in {}: {}'.format(func.__name__, args))
            logger.debug('Passed kwargs in {}: {}'.format(func.__name__, kwargs))
            return func(*args, **kwargs)
        return wrapper
    return logger_decorator


def log_outputs_into(logger):

    def logger_decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            ret_val = func(*args, **kwargs)
            logger.debug('Computed outputs in {}: {}'.format(func.__name__, ret_val))
            return ret_val
        return wrapper
    return logger_decorator

