import mxnet as mx
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


def parse_network(network, outputs, inputs, pretrained=False, ctx=mx.cpu()):
    """Parse network with specified outputs and other arguments.

    Parameters
    ----------
    network : str or HybridBlock or Symbol
        Logic chain: load from gluoncv.model_zoo if network is string.
        Convert to Symbol if network is HybridBlock
    outputs : str or iterable of str
        The name of layers to be extracted as features.
    inputs : iterable of str
        The name of input datas.
    pretrained : bool
        Use pretrained parameters as in gluon.model_zoo

    Returns
    -------
    inputs : list of Symbol
        Network input Symbols, usually ['data']
    outputs : list of Symbol
        Network output Symbols, usually as features
    params : ParameterDict
        Network parameters.
    """

    import mxnet as mx
    from mxnet.symbol import Symbol
    from mxnet.gluon import HybridBlock
    from mxnet.base import string_types

    inputs = list(inputs) if isinstance(inputs, tuple) else inputs
    for i, inp in enumerate(inputs):
        if isinstance(inp, string_types):
            inputs[i] = mx.sym.var(inp)
        assert isinstance(inputs[i], Symbol), "Network expects inputs are Symbols."
    if len(inputs) == 1:
        inputs = inputs[0]
    else:
        inputs = mx.sym.Group(inputs)
    params = None
    prefix = ''
    if isinstance(network, string_types):
        from gluoncv.model_zoo import get_model
        network = get_model(network, pretrained=pretrained, ctx=ctx)
    if isinstance(network, HybridBlock):
        params = network.collect_params()
        prefix = network._prefix
        network = network(inputs)
    assert isinstance(network, Symbol), \
        "FeatureExtractor requires the network argument to be either " \
        "str, HybridBlock or Symbol, but got %s"%type(network)

    if isinstance(outputs, string_types):
        outputs = [outputs]
    assert len(outputs) > 0, "At least one outputs must be specified."
    outputs = [out if out.endswith('_output') else out + '_output' for out in outputs]
    outputs = [network.get_internals()[prefix + out] for out in outputs]
    return inputs, outputs, params


def symbol_to_symbol_block(prefix, epoch, inputs, outputs):
    import mxnet.gluon as gluon
    from kungfutils.routines.string_routines import generate_id

    sym, arg_params, aux_params = mx.model.load_checkpoint(prefix=prefix, epoch=epoch)

    inp_names = inputs
    inputs = mx.sym.Group(*[mx.sym.var(i) for i in inputs])
    sym = sym(inputs)

    internals = sym.get_internals()
    outputs = [internals['{}_output'.format(i)] for i in outputs]
    outputs = mx.sym.Group(outputs)

    model_uid = generate_id(10)

    mx.model.save_checkpoint(
        prefix=model_uid, epoch=0,
        arg_params=arg_params, aux_params=aux_params, symbol=outputs)

    block = gluon.SymbolBlock.imports(
        '{}-symbol.json'.format(model_uid), ctx=mx.cpu(), input_names=inp_names)
    block.load_parameters(
        '{}-0000.params'.format(model_uid), allow_missing=True, ignore_extra=True, ctx=mx.cpu())

    return block
