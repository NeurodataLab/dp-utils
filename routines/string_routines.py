import re
import logging

from .. import ROOT_LOGGER_NAME, ROOT_LOGGER_LEVEL
logger = logging.getLogger('{}.{}'.format(ROOT_LOGGER_NAME, __name__))
logger.setLevel(ROOT_LOGGER_LEVEL)


def with_regex_precompilation(**rgx_dict):
    regexps = {k: re.compile(rgx_string) for k, rgx_string in rgx_dict.items()}

    def wrapper(func):
        def wrapped(*args, **kwargs):
            kwargs.update(regexps)
            return func(*args, **kwargs)
        return wrapped
    return wrapper


@with_regex_precompilation(format_rgx=r"{}")
def format_string_match(format_string, result, **kwargs):
    format_rgx = kwargs['format_rgx']
    format_rgxp = format_rgx.sub(string=format_string, repl="(.+)")

    findings = re.findall(pattern=format_rgxp, string=result)
    if len(findings) > 1:
        logger.debug('number of matches is more than 1 for: {} and {}'.format(format_string, result))
    return findings[0]


def is_overstring_of_any(str, lst):
    ret = False
    for to_cmp in lst:
        ret = ret or (to_cmp in str)
    return ret


def is_substring_of_any(str, lst):
    ret = False
    for to_cmp in lst:
        ret = ret or (str in to_cmp)
    return ret


if __name__ == '__main__':
    print format_string_match(
        format_string='/home/mininlab/DATA/EM/faces/{}/{}/f_imgs',
        result='/home/mininlab/DATA/EM/faces/_10_Questions2/fragment_3/f_imgs'
    )