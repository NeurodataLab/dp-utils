from collections import defaultdict

import networkx as nx

from .base_preprocessor import BasePreprocessor


class CompositePreprocessor(BasePreprocessor):
    def __init__(self, input_names, output_names=None, *args, **kwargs):
        super(CompositePreprocessor, self).__init__(*args, **kwargs)
        self._proc_graph = nx.MultiDiGraph()

        self._input_nodes = {'input_{}'.format(name): name for name in input_names}
        self._graph_heads = {name: 'input_{}'.format(name) for name in input_names}
        self._nodes_ready = {}
        self._processors = {}

        self._output_names = output_names

        for num, input_name in enumerate(input_names):
            self._proc_graph.add_node('input_{}'.format(input_name), name=input_name)

    def process(self, **kwargs):
        assert len(kwargs) == len(self._input_nodes), 'Not all inputs can be consumed'
        available_data = defaultdict(dict)

        comp_queue = self._input_nodes.keys()
        comp_state = {k: kwargs[k] for k in self._input_nodes.values()}

        while len(comp_queue) != 0:
            node_id = comp_queue.pop(0)
            if node_id in self._processors:
                processor = self._proc_graph.node[node_id]['processor']
                inp = {name: comp_state[name] for name in processor.provide_input}
                comp_state.update(processor.process(**inp))

            for successor in self._proc_graph.successors(node_id):
                passed_args = self._proc_graph.get_edge_data(node_id, successor)
                successor_processor = self._proc_graph.node[successor]['processor']
                if successor not in available_data:
                    for name in successor_processor.provide_input:
                        available_data[successor][name] = False

                for edge_dict in passed_args.values():
                    arg_name = edge_dict['arg_name']
                    available_data[successor][arg_name] = True

                if all(available_data[successor].values()):
                    comp_queue.append(successor)

        return comp_state if self._output_names is None else {name: comp_state[name] for name in self._output_names}

    def add(self, processor, **kwargs):
        """
        :param processor: object with method process
        """
        proc_name = kwargs.get('name', processor.__class__.__name__)
        if proc_name in self._processors:
            self.add(processor, name='{}_0'.format(proc_name))
            return

        self._processors[proc_name] = processor
        self._proc_graph.add_node(proc_name, processor=processor, name=proc_name)

        for name in processor.provide_input:
            assert name in self._graph_heads, 'No {} in graph heads at the moment'.format(name)
            self._proc_graph.add_edge(self._graph_heads[name], proc_name, arg_name=name)
        self._graph_heads.update({new_out_name: proc_name for new_out_name in processor.provide_output})

    @property
    def provide_data(self):
        cum_info = {}
        for name, proc_name in self._graph_heads.items():
            if proc_name in self._processors:
                cum_info.update({k: v for k, v in self._processors[proc_name].provide_data
                                 if k in self._output_names})

        if self._output_names is None:
            return cum_info.items()
        else:
            return [(name, cum_info[name]) for name in self._output_names]

    @property
    def provide_output(self):
        return self._output_names or self._graph_heads.keys()

    @property
    def provide_input(self):
        return self._input_nodes.values()
