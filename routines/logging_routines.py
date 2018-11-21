from functools import wraps


def log_args_into(logger):
    def logger_decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger.info('Passed args: {}'.format(args))
            logger.info('Passed kwargs: {}'.format(kwargs))
            return func(*args, **kwargs)
        return wrapper
    return logger_decorator
