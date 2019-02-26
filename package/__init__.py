import logging
ROOT_LOGGER_NAME = ''
ROOT_LOGGER_LEVEL = logging.INFO


def set_logger_level(level):
    global ROOT_LOGGER_LEVEL
    ROOT_LOGGER_LEVEL = level


def set_logger_name(name):
    global ROOT_LOGGER_NAME
    ROOT_LOGGER_NAME = name
