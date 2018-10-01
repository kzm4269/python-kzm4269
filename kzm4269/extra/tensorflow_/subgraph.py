import copy
import itertools as it
import re
from collections import defaultdict
from pathlib import PurePosixPath as _Path

import tensorflow as tf
from google.protobuf import text_format
from tensorflow.contrib.framework.python.ops import get_graph_from_inputs
from tensorflow.python.framework import meta_graph as meta_graph_lib

__all__ = [
    'extract_subgraph',
    'StaticGraph',
    'custom_object_scope',
    'as_layer',
    'layer_type',
    'unique_layer_name',
]


def _as_list(obj) -> list:
    if isinstance(obj, str):
        return [obj]
    try:
        return list(obj)
    except (TypeError, ValueError):
        return [obj]


def _as_graph(obj):
    if obj is None:
        graph = tf.get_default_graph()
    elif isinstance(obj, tf.Graph):
        graph = obj
    elif isinstance(obj, tf.GraphDef):
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(obj)
    else:
        with tf.Graph().as_default() as graph:
            meta_graph_lib.import_scoped_meta_graph(obj)
    return graph


def _as_op_name(obj):
    if isinstance(obj, tf.Operation):
        return obj.name
    elif isinstance(obj, tf.Variable):
        return obj.value().op.name
    elif isinstance(getattr(obj, 'op', None), tf.Operation):
        return obj.op.name
    elif isinstance(obj, bytes):
        return obj.decode().split('@')[1]
    elif isinstance(obj, str):
        return obj.lstrip('^').split(':')[0]
    else:
        raise TypeError(obj)


def _as_tensor_name(obj):
    if isinstance(obj, tf.Operation):
        output, = obj.outputs
        return output.name
    elif isinstance(obj, tf.Variable):
        return obj.value().name
    elif isinstance(obj, tf.Tensor):
        return obj.name
    elif isinstance(obj, str):
        return (obj if ':' in obj else obj + ':0').lstrip('^')
    else:
        raise TypeError(obj)


def _join_scope(*scopes):
    return str(_Path(*scopes)).lstrip('.')


def _rel_scope(scope, relative_to):
    return str(_Path('/' + scope).relative_to('/' + relative_to)).lstrip('/')


def _common_scope(*scopes):
    return '/'.join(
        part
        for part, in it.takewhile(
            lambda parts: len(parts) == 1,
            map(set, zip(*(_Path(scope).parts for scope in scopes)))
        )
    )


def _parent_scope(scope):
    return '/'.join(_Path(scope).parts[:-1])


def _base_op_name(name):
    return _Path(name).name


def unique_layer_name(name):
    return type(name, (tf.keras.layers.Layer,), {})().name


def _pascal_case(name):
    return ''.join(
        word[0].upper() + word[1:]
        for word in re.split(r'([^a-zA-Z])', name)
        if re.match(r'[a-zA-Z0-9]+$', word)
    )


def _extract_valid_collections(
        graph_def, graph, import_kw=None, export_kw=None):
    """Remove collections which cause a error when importing.

    Parameters
    ----------
    graph_def : tf.GraphDef
    graph : tf.Graph
    import_kw : dict (optional)
    export_kw : dict (optional)

    Returns
    -------
    meta_graph_def : tf.MetaGraphDef
    accepted_collections : dict
    """
    import_kw = dict(import_kw or {})
    export_kw = dict(export_kw or {})

    # Save original collections.
    collections = {
        key: graph.get_collection(key)
        for key in graph.get_all_collection_keys()
    }

    try:
        # Clear all collections.
        for key in collections:
            graph.clear_collection(key)

        # Create the graph which has only nodes and no collections.
        with tf.Graph().as_default() as test_graph:
            tf.import_graph_def(graph_def, name='import', **import_kw)
            import_kw.pop('input_map', None)

        # Add collections one at a time to find the cause of errors.
        accepted_collections = {key: [] for key in collections}
        for key, collection in collections.items():
            for item in collection:
                # Create MetaGraphDef which has only one collection.
                graph.add_to_collection(key, item)
                item_def, _ = meta_graph_lib.export_scoped_meta_graph(
                    graph=graph,
                    graph_def=graph_def,
                )

                # It is easier to ask forgiveness than permission.
                try:
                    meta_graph_lib.import_scoped_meta_graph(
                        item_def,
                        import_scope='import',
                        graph=test_graph,
                    )
                except (ValueError, KeyError):
                    pass
                else:
                    accepted_collections[key].append(item)

                graph.clear_collection(key)

        # Restore accepted collections.
        for key, collection in accepted_collections.items():
            for item in collection:
                graph.add_to_collection(key, item)

        # Export result.
        meta_graph_def, _ = meta_graph_lib.export_scoped_meta_graph(
            graph=graph,
            graph_def=graph_def,
            **export_kw,
        )
    finally:
        # Restore all collections.
        for key, collection in collections.items():
            graph.clear_collection(key)
            for item in collection:
                graph.add_to_collection(key, item)

    return meta_graph_def, accepted_collections


def _extract_subgraph(input_ts, output_ts, graph, extra_dependencies=None):
    """Extract the subgraph that can reach any of the output tensors.

    Parameters
    ----------
    output_ts : sequence of tf.Tensor
    input_ts : sequence of tf.Tensor
    graph : tf.Graph
    extra_dependencies : dict

    Returns
    -------
    meta_graph_def : tf.MetaGraphDef
    collections : dict

    Examples
    --------
    >>> graph = tf.get_default_graph()
    >>> x = tf.placeholder(tf.float32, [3], 'x')
    >>> v = tf.Variable(1., name='v')
    >>> y = x + v
    >>> meta_graph_def, collections = _extract_subgraph([x], [y], graph)
    >>> [node.name for node in meta_graph_def.graph_def.node]
    ['x', 'v', 'v/read', 'add']
    """
    graph_def = graph.as_graph_def(add_shapes=True)
    _modified = False

    # Replace each input tensor by a placeholder.
    with tf.Graph().as_default():
        placeholder_from_op_name = {
            t.op.name: tf.placeholder(t.dtype, t.shape, t.op.name)
            for t in input_ts
            if t.op.type != 'Placeholder'
        }
    for node in graph_def.node:
        if node.name in placeholder_from_op_name:
            node.CopyFrom(placeholder_from_op_name[node.name].op.node_def)
            _modified = True

    # Find the direct inputs of each node.
    inputs_from_name = defaultdict(set)
    inputs_from_name.update({
        node.name: {
            _as_op_name(name)
            for name in node.input
        }
        for node in graph_def.node
    })

    # Add colocation nodes.
    for node in graph_def.node:
        inputs_from_name[node.name] |= set(map(
            _as_op_name,
            node.attr['_class'].list.s,
        ))

    # Add extra dependencies.
    if extra_dependencies:
        for name, deps in extra_dependencies.items():
            inputs_from_name[name] |= set(deps)

    # Resolve dependencies.
    stack = [t.op.name for t in output_ts]
    depended_names = {t.op.name for t in input_ts}
    while stack:
        top = stack.pop()
        if top not in depended_names:
            depended_names.add(top)
            stack.extend(inputs_from_name[top])

    # Delete nodes that are not dependent on the output nodes.
    for i, node in reversed(list(enumerate(graph_def.node))):
        if node.name not in depended_names:
            if node.op != 'Exit':
                del graph_def.node[i]
                _modified = True

    if _modified:
        meta_graph_def, collections = _extract_valid_collections(
            graph_def=graph_def,
            graph=graph,
        )
    else:
        meta_graph_def, _ = meta_graph_lib.export_scoped_meta_graph(
            graph=graph,
        )
        collections = {
            key: graph.get_collection(key)
            for key in graph.get_all_collection_keys()
        }

    return meta_graph_def, collections


def _fix_node_attrs(meta_graph_def, import_scope):
    meta_graph_def = copy.deepcopy(meta_graph_def)
    targets = ['frame_name']

    import_scope = tf.get_default_graph().unique_name(
        name=import_scope,
        mark_as_used=False,
    )

    for node in meta_graph_def.graph_def.node:
        for key in targets:
            if key in node.attr:
                s = node.attr[key].s
                s = _join_scope(import_scope, s.decode())
                node.attr[key].s = s.encode()

    return meta_graph_def


def extract_subgraph(
        inputs=None, outputs=None, graph=None,
        unwrap_scope=False, unbind_inputs=False,
        keep_variables=True):
    """Extract the subgraph that can reach any of the output tensors.

    Parameters
    ----------
    inputs : iterable of the graph elements or their names
    outputs : iterable of the graph elements or their names
        If no outputs are given, use all the tensors which has no output ops.
    graph : tf.Graph, tf.GraphDef, tf.MetaGraphDef or path_like (optional)
        If graph is None (default), use the default graph.
    unwrap_scope : bool or str (optional)
        If `unwrap_scope` is true, remove the common name scope for all ops.
        If `unwrap_scope` is a string, use it as the common name scope.
    unbind_inputs : bool or str (optional)
        If `unbind_inputs` is false (default), replace each input tensor by a
        placeholder.  If true is given, remove input tensors from the
        MetaGraphDef.  In this case, you will need the input_map when
        importing the result.  If a string is given, use it as the
        `unbind_inputs_col_name` keyword argument when exporing.
    keep_variables : bool or variables (optional)
        If true is given (default), make operations of a variable are depends
        on each other.  If a iterable of variables or their names are given,
        the above processing is performed only for these.

    Returns
    -------
    meta_graph_def : tf.MetaGraphDef
    unwrapped_scope : str
    var_list : dict
        Mapping from the renamed variable names to original variables.

    Examples
    --------
    >>> tf.reset_default_graph()
    >>> x = tf.placeholder(tf.float32, [3], 'x')
    >>> v = tf.Variable(1., name='v')
    >>> y = x + v
    >>> meta_graph_def, unwrapped_scope, var_list = extract_subgraph(x, y)
    >>> [node.name for node in meta_graph_def.graph_def.node]
    ['x', 'v/initial_value', 'v', 'v/Assign', 'v/read', 'add']
    >>> list(var_list.keys())
    ['v:0']

    >>> tf.reset_default_graph()
    >>> x = tf.placeholder(tf.float32, [None, 3], 'x')
    >>> y = tf.keras.layers.Dense(5)(x)
    >>> meta_graph_def, unwrapped_scope, var_list = extract_subgraph(x, y)
    >>> list(var_list.keys())
    ['dense/kernel:0', 'dense/bias:0']
    """
    # Parse argumetns.
    given_graph = graph
    input_list = [] if inputs is None else _as_list(inputs)
    output_list = [] if outputs is None else _as_list(outputs)
    graph = get_graph_from_inputs(input_list + output_list, _as_graph(graph))

    input_ts = [
        graph.get_tensor_by_name(_as_tensor_name(t))
        for t in input_list
    ]

    if not output_list:
        # Find all tensors which has no output ops.
        all_ts = {
            output_tensor
            for op in graph.get_operations()
            for output_tensor in op.outputs
        }
        tensors_which_has_output_ops = {
            input_tensor
            for op in graph.get_operations()
            for input_tensor in op.inputs
        }
        output_ts = list(all_ts - tensors_which_has_output_ops)
    else:
        output_ts = [
            graph.get_tensor_by_name(_as_tensor_name(t))
            for t in output_list
        ]

    # Setup extra dependencies.
    extra_deps = defaultdict(set)
    if keep_variables:
        if keep_variables is True:
            with graph.as_default():
                variables = tf.global_variables()
        else:
            variables = _as_list(keep_variables)
            with get_graph_from_inputs(variables, graph).as_default():
                var_from_name = {v.name: v for v in tf.global_variables()}
                for i, v in enumerate(variables[:]):
                    if isinstance(v, str):
                        variables[i] = var_from_name[v]
                    elif not isinstance(v, tf.Variable):
                        raise ValueError(
                            '`keep_variables` must be bool or variables, not '
                            + repr(type(v).__name__)
                        )
        for v in variables:
            for attr in vars(v).values():
                if isinstance(attr, (tf.Tensor, tf.Operation)):
                    extra_deps[v.op.name].add(_as_op_name(attr))

    def _imported(name):
        return _join_scope(import_scope, name)

    if unbind_inputs:
        # Get a unique name scope.
        import_scope = graph.unique_name('import', mark_as_used=False)

        # Move the operations other than the input tensors to import_scope.
        meta_graph_def, _ = meta_graph_lib.export_scoped_meta_graph(
            graph=graph)

        with tf.Graph().as_default() as g:
            input_map = {
                t.op.name: tf.placeholder(t.dtype, t.shape, t.op.name)
                for t in input_ts
            }
            meta_graph_lib.import_scoped_meta_graph(
                meta_graph_def,
                import_scope=import_scope,
                input_map=input_map,
            )
            meta_graph_def, collections = _extract_subgraph(
                graph=g,
                input_ts=input_map.values(),
                output_ts=[
                    g.get_tensor_by_name(_imported(t.name))
                    for t in output_ts
                ],
                extra_dependencies={
                    _imported(name): list(map(_imported, deps))
                    for name, deps in extra_deps.items()
                })
    else:
        import_scope = ''
        meta_graph_def, collections = _extract_subgraph(
            graph=graph,
            input_ts=input_ts,
            output_ts=output_ts,
            extra_dependencies=extra_deps)

    # Find the common name scope as `export_scope`.
    if unwrap_scope:
        if unbind_inputs:
            excludes = {t.op.name for t in input_ts}
            excludes |= {_imported(t.op.name) for t in input_ts}
        else:
            excludes = ()

        common_scope = _common_scope(*(
            _parent_scope(node.name)
            for node in meta_graph_def.graph_def.node
            if _parent_scope(node.name) and node.name not in excludes
        ))

        # If unwrap_scope is a string, use it as common_scope.
        if isinstance(unwrap_scope, str):
            _rel_scope(common_scope, unwrap_scope)  # Assert success.
            common_scope = unwrap_scope

        export_scope = common_scope
    else:
        export_scope = import_scope

    # Extract only the operations in `export_ecope`.
    if unwrap_scope or unbind_inputs:
        if isinstance(unbind_inputs, str):
            unbound_inputs_col_name = unbind_inputs
        else:
            unbound_inputs_col_name = 'unbound_input'

        with tf.Graph().as_default():
            meta_graph_lib.import_scoped_meta_graph(meta_graph_def)
            meta_graph_def, _ = meta_graph_lib.export_scoped_meta_graph(
                export_scope=export_scope,
                unbound_inputs_col_name=unbound_inputs_col_name,
            )

        # There is a possibility that the unbound inputs collection are
        # duplicated. It is probably a bug. Deduplicate the collection.
        collection = meta_graph_def.collection_def[unbound_inputs_col_name]
        unbound_inputs = collection.bytes_list.value
        seen = set()
        for i, item in list(enumerate(unbound_inputs))[::-1]:
            if item in seen:
                del unbound_inputs[i]
            seen.add(item)
        if not len(unbound_inputs):
            del meta_graph_def.collection_def[unbound_inputs_col_name]

    # Make result.
    if import_scope == export_scope:
        unwrapped_scope = ''
    else:
        unwrapped_scope = _rel_scope(export_scope, import_scope)

    var_list = {
        _rel_scope(v.name, unwrapped_scope): v
        for v in collections.get(tf.GraphKeys.GLOBAL_VARIABLES, ())
    }
    if graph is not given_graph:
        var_list = {name: v.name for name, v in var_list.items()}

    return meta_graph_def, unwrapped_scope, var_list


class StaticGraph(tf.keras.layers.Layer):
    """Layer based on static MetaGraphDef"""

    def __init__(
            self, inputs, outputs, *, training=None, updates=None,
            losses=None, graph=None, default_name=None, **kwargs):
        # Verify arguments.
        input_list = _as_list(inputs)
        output_list = _as_list(outputs)
        training_list = [] if training is None else _as_list(training)
        update_list = [] if updates is None else _as_list(updates)
        loss_list = [] if losses is None else _as_list(losses)

        if isinstance(graph, tf.MetaGraphDef):
            meta_graph_def = graph
            with tf.Graph().as_default() as graph:
                meta_graph_lib.import_scoped_meta_graph(meta_graph_def)

        graph = get_graph_from_inputs(
            op_input_list=(
                    input_list
                    + output_list
                    + training_list
                    + update_list
                    + loss_list
            ),
            graph=graph,
        )

        if not training_list:
            with graph.as_default():
                training_list.append(tf.keras.backend.learning_phase())

        self.__inputs = list(map(_as_tensor_name, input_list))
        self.__outputs = list(map(_as_tensor_name, output_list))
        self.__training = list(map(_as_tensor_name, training_list))
        self.__updates = list(map(_as_op_name, update_list))
        self.__losses = list(map(_as_tensor_name, loss_list))

        # Determine whether the output tensors depends on the learning phase.
        if self.__training:
            def _uses_lp(t, path=(), targets=self.__training):
                return t.name in targets or t.name not in path and any(
                    _uses_lp(ti, path + (t.name,))
                    for ti in t.op.inputs
                )

            self.__uses_lp = {
                name: _uses_lp(graph.get_tensor_by_name(name))
                for name in self.__outputs
            }

            if not any(self.__uses_lp.values()):
                self.__training = []
        else:
            self.__uses_lp = {name: False for name in self.__outputs}

        # Total inputs and outputs are confirmed at this point.
        total_inputs = self.__inputs + self.__training
        total_outputs = self.__outputs + self.__updates + self.__losses

        io_conflict = set(total_inputs) & set(total_outputs)
        if io_conflict:
            raise ValueError(
                'Output tensors contains input tensors: ' + str(io_conflict))

        # Extract the minimum MetaGraphDef required for calculation.
        (
            self.__meta_graph_def,
            _,
            var_from_name,
        ) = extract_subgraph(
            graph=graph,
            inputs=total_inputs,
            outputs=total_outputs,
        )

        # Extract oprations other then the input tensors and the variables.
        (
            self.__unbound_meta_graph_def,
            self.__export_scope,
            _,
        ) = extract_subgraph(
            graph=graph,
            inputs=total_inputs,
            outputs=total_outputs,
            unwrap_scope=True,
            unbind_inputs=True,
            keep_variables=False,
        )

        # Variables are not built, yet.
        self.__var_from_name = {}

        # Use the common name scope as the default name.
        if default_name is None and self.__export_scope:
            default_name = _base_op_name(self.__export_scope)
        if kwargs.get('name') is None and default_name is not None:
            kwargs['name'] = unique_layer_name(default_name)

        super().__init__(**kwargs)

        if not var_from_name:
            self.built = True

    def build(self, input_shape):
        # Extract and import only the variables.
        with tf.Graph().as_default():
            meta_graph_lib.import_scoped_meta_graph(self.__meta_graph_def)
            assert tf.global_variables()
            variables_def, var_export_scope, _ = extract_subgraph(
                outputs=tf.global_variables(),
                unwrap_scope=self.__export_scope,
            )

        var_from_name = meta_graph_lib.import_scoped_meta_graph(
            variables_def,
            import_scope='variables',
        )

        # Update attribudes.
        trainable_variable_names = {v.name for v in tf.trainable_variables()}
        for v in sorted(var_from_name.values(), key=lambda v: v.name):
            if v.name in trainable_variable_names:
                self._trainable_weights.append(v)
            else:
                self._non_trainable_weights.append(v)

        self.__var_from_name = {
            _join_scope(var_export_scope, name): v.op.outputs[0]
            for name, v in var_from_name.items()
        }

        self.built = True

    def call(self, inputs, training=None):
        input_list = _as_list(inputs)
        graph = get_graph_from_inputs(input_list)
        if training is None:
            training = tf.keras.backend.learning_phase()

        if len(self.__inputs) > len(input_list):
            raise ValueError('missing {} required arguments: {}'.format(
                len(self.__inputs) - len(input_list),
                list(self.__inputs[len(input_list):]),
            ))
        elif len(self.__inputs) < len(input_list):
            raise ValueError('{} extra arguments were given'.format(
                len(input_list) - len(self.__inputs),
            ))

        input_map = {
            _rel_scope(name, self.__export_scope): v.op.outputs[0]
            for name, v in self.__var_from_name.items()
        }
        input_map.update({
            '$unbound_inputs_' + name: input_tensor
            for name, input_tensor in zip(self.__inputs, input_list)
        })
        input_map.update({
            '$unbound_inputs_' + name: training
            for name in self.__training
        })

        # Import the unbound graph.
        import_scope = tf.get_default_graph().unique_name(
            name='call',
            mark_as_used=False,
        )
        meta_graph_lib.import_scoped_meta_graph(
            _fix_node_attrs(
                self.__unbound_meta_graph_def,
                import_scope=_base_op_name(import_scope)),
            import_scope=_base_op_name(import_scope),
            input_map=input_map,
        )

        def _import_scope(name):
            return _join_scope(
                import_scope,
                _rel_scope(name, self.__export_scope),
            )

        # Save training operations.
        self.add_update(
            inputs=inputs,
            updates=[
                graph.get_operation_by_name(_import_scope(name))
                for name in self.__updates
            ],
        )

        self.add_loss(
            inputs=inputs,
            losses=[
                graph.get_tensor_by_name(_import_scope(name))
                for name in self.__losses
            ],
        )

        # Find the output tensors and set protected attribudes.
        outputs = [
            graph.get_tensor_by_name(_import_scope(name))
            for name in self.__outputs
        ]

        if training is tf.keras.backend.learning_phase():
            for name, t in zip(self.__outputs, outputs):
                setattr(t, '_uses_learning_phase', self.__uses_lp[name])

        return outputs if len(outputs) >= 2 else outputs[0]

    def compute_mask(self, inputs, mask=None):
        super().compute_mask(inputs, mask)
        assert mask is None or all(m is None for m in mask), mask
        if len(self.__outputs) >= 2:
            return [None] * len(self.__outputs)
        else:
            return None

    def get_config(self):
        config = {
            'inputs': self.__inputs,
            'outputs': self.__outputs,
            'training': self.__training,
            'updates': self.__updates,
            'losses': self.__losses,
            'graph': text_format.MessageToString(self.__meta_graph_def),
        }

        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        meta_graph_def = tf.MetaGraphDef()
        text_format.Merge(config.pop('graph'), meta_graph_def)
        return cls(graph=meta_graph_def, **config)

    @property
    def weight_dict(self):
        return {
            _as_op_name(_rel_scope(name, self.__export_scope)): value
            for name, value in self.__var_from_name.items()
        }


def custom_object_scope():
    return tf.keras.utils.custom_object_scope({
        StaticGraph.__name__: StaticGraph,
    })


def as_layer(
        inputs, outputs, *, training=None, updates=None, losses=None,
        graph=None, default_name=None, **kwargs):
    """Create a Keras layer from a constructed graph.

    This function extracts the subgraph from the input tensors to the output
    tensors as a Keras layer.

    Parameters
    ----------
    inputs : iterable of tf.Tensor or their names
        Input tensors of the subgraph.
    outputs : iterable of tf.Tensor, their names, or callable
        Output tensors of the subgraph. If `output` is callable, use its return
        value as the output tensors.
    training : the learning phase tensor or their names (optional)
    updates : iterable of tf.Operation or their names (optional)
    losses : iterable of tf.Tensor or their names (optional)
    graph : tf.Graph or tf.MetaGraphDef (optional)
        Target graph to be extracted.
    default_name : str (optional)
    kwargs

    Returns
    -------
    layer : StaticGraph

    Examples
    --------
    This function extracts the specified subgraph as a layer.
    >>> x = tf.keras.Input(shape=[3])
    >>> layer = as_layer(x, 2 * x + 1)
    >>> y = layer(x)
    >>> model = tf.keras.models.Model(x, y)
    >>>
    >>> # You should use `custom_objects_scope()` when loading the model.
    >>> model.save('/tmp/model.h5')
    >>> with custom_object_scope():
    ...     model = tf.keras.models.load_model('/tmp/model.h5')

    If the subgraph contains the inner layers which has `updates` or `losses`
    attribudes such as `BatchNormalization`, you should pass these as arguments
    expressly.
    >>> x = tf.placeholder(dtype=tf.float32, shape=(None, 3))
    >>> bn = tf.keras.layers.BatchNormalization()
    >>> y = bn(x)
    >>> layer = as_layer(x, y, updates=bn.updates)
    """
    return StaticGraph(
        inputs, outputs,
        training=training,
        updates=updates,
        losses=losses,
        graph=graph,
        default_name=default_name,
        **kwargs,
    )


def layer_type(op_fn, name=None, *, new_graph=True, pack_args=False):
    """Create a type which builds a layer using `op_fn`.

    Parameters
    ----------
    op_fn : callable
        Function that, given input tensors and parameters, creates a part of
        the graph and returns output tensors.
    name : str (optional)
        If `name` is None (default), use the name of `op_fn`.
    new_graph : bool or tf.Graph (optional)
        If `new_graph` is true (default), use a new `tf.Graph` as the default
        graph when calling `call_fun`. If `new_graph` is a `tf.Graph`, use it
        as the default graph.
    pack_args : bool (optional)
        If true is given, pass the input tensors as a list to `op_fun`.

    Returns
    -------
    layer_type : type

    Examples
    --------
    The return value is not a subclass of `tf.keras.layers.Layer`, but it can be
    used as well.
    >>> Reshape = layer_type(tf.reshape)
    >>> Add = layer_type(tf.add)
    >>> Concat = layer_type(tf.concat, pack_args=True)
    >>>
    >>> x = tf.keras.Input(shape=[6])
    >>> y = Reshape((-1, 2, 3))(x)
    >>> y = Add()([y, y])
    >>> y = Concat(axis=1)([y, y])
    >>> model = tf.keras.models.Model(x, y)
    >>>
    >>> # You should use `custom_objects=` when loading the model.
    >>> model.save('/tmp/model.h5')
    >>> with custom_object_scope():
    ...     model = tf.keras.models.load_model('/tmp/model.h5')
    """
    if name is None:
        try:
            name = op_fn.__name__
        except AttributeError:
            raise ValueError('specify name of type')

    if new_graph is not None and not isinstance(new_graph, (bool, tf.Graph)):
        raise ValueError(
            '`new_graph` must be a bool or a tf.Graph, not '
            + repr(type(new_graph).__name__)
        )

    class LazyStaticGraph:
        def __init__(self, *args, **kwargs):
            layer_kwargs = {
                'activity_regularizer',
                'batch_size',
                'name',
                'trainable',
                'weights',
            }
            self._init_kwargs = dict(
                default_name=name)
            for key in list(kwargs):
                if key in layer_kwargs:
                    self._init_kwargs[key] = kwargs.pop(key)

            self._op_args = args
            self._op_kwargs = kwargs
            self._layer = None

        def __call__(self, inputs, **kwargs):
            base_inputs = _as_list(inputs)

            if self._layer is None:
                base_graph = get_graph_from_inputs(base_inputs)

                if new_graph is None or new_graph is False:
                    graph = base_graph
                elif new_graph is True:
                    graph = tf.Graph()
                elif isinstance(new_graph, tf.Graph):
                    graph = new_graph
                else:
                    assert False, new_graph

                with graph.as_default():
                    input_ts = [
                        t if t.graph is graph
                        else tf.placeholder(t.dtype, t.shape, t.op.name)
                        for t in map(tf.convert_to_tensor, base_inputs)
                    ]
                    output_ts = op_fn(
                        *([input_ts] if pack_args else input_ts),
                        *self._op_args,
                        **self._op_kwargs,
                    )

                init_kwargs = dict(self._init_kwargs)
                if isinstance(output_ts, dict):
                    init_kwargs.update(output_ts)
                else:
                    init_kwargs.update(outputs=output_ts)

                self._layer = StaticGraph(input_ts, **init_kwargs)

            return self._layer(base_inputs, **kwargs)

        @property
        def built_layer(self):
            if self._layer is None:
                raise AttributeError('not initialized')
            return self._layer

    return type(_pascal_case(name), (), {
        '__init__': LazyStaticGraph.__init__,
        '__call__': LazyStaticGraph.__call__,
        'built_layer': LazyStaticGraph.built_layer,
    })
