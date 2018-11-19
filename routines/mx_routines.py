import json
import copy
import logging

from .. import ROOT_LOGGER_NAME, ROOT_LOGGER_LEVEL
logger = logging.getLogger('{}.{}'.format(ROOT_LOGGER_NAME, __name__))
logger.setLevel(ROOT_LOGGER_LEVEL)


def rename_params(params, prefix):
    renamed_params = {}
    for name, value in params.items():
        if prefix in name:
            raise Exception("Existing prefixed name found while renaming")
        renamed_params['{}/{}'.format(prefix, name)] = value
    return renamed_params


def rename_symbol(symbol_path, prefix):
    symbol_data = json.load(open(symbol_path))

    renamed_symbol_data = copy.copy(symbol_data)
    for block in renamed_symbol_data['nodes']:
        name = block['name']
        if prefix in name:
            logger.warning('Symbol {} is already renamed, no renaming performed'.format(symbol_path))
            return
        block.update({'name': '{}/{}'.format(prefix, name)})
    json.dump(renamed_symbol_data, open(symbol_path, 'w'))
    json.dump(symbol_data, open(symbol_path + '.original', 'w'))
